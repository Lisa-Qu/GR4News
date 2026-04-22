"""Cache backends for the backend-first service skeleton."""

from __future__ import annotations

from dataclasses import dataclass
import socket
import ssl
from threading import Lock
from time import monotonic
from typing import Callable, Generic, Protocol, TypeVar
from urllib.parse import unquote, urlparse


ValueT = TypeVar("ValueT")
SerializedT = TypeVar("SerializedT", bound=str)


class CacheBackend(Protocol[ValueT]):
    """Minimal cache interface used by the retrieval service."""

    def get(self, key: str) -> ValueT | None:
        """Return cached value or None."""

    def set(self, key: str, value: ValueT) -> None:
        """Store one cache value."""

    def clear(self) -> None:
        """Drop all cache entries when supported."""


@dataclass
class _CacheEntry(Generic[ValueT]):
    value: ValueT
    expires_at: float


class TTLCache(Generic[ValueT]):
    """Thread-safe in-memory TTL cache.

    This is sufficient for the first backend skeleton and can later be swapped
    for Redis without changing the service interface.
    """

    def __init__(self, ttl_seconds: int = 600) -> None:
        self._ttl_seconds = ttl_seconds
        self._entries: dict[str, _CacheEntry[ValueT]] = {}
        self._lock = Lock()

    def get(self, key: str) -> ValueT | None:
        now = monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                self._entries.pop(key, None)
                return None
            return entry.value

    def set(self, key: str, value: ValueT) -> None:
        expires_at = monotonic() + self._ttl_seconds
        with self._lock:
            self._entries[key] = _CacheEntry(value=value, expires_at=expires_at)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class ResilientCache(Generic[ValueT]):
    """Cache wrapper that falls back when the primary backend is unavailable."""

    def __init__(
        self,
        primary: CacheBackend[ValueT],
        fallback: CacheBackend[ValueT],
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._primary_available = True
        self._lock = Lock()

    def get(self, key: str) -> ValueT | None:
        primary = self._get_primary()
        if primary is not None:
            try:
                value = primary.get(key)
                if value is not None:
                    self._fallback.set(key, value)
                return value
            except Exception:
                self._disable_primary()
        return self._fallback.get(key)

    def set(self, key: str, value: ValueT) -> None:
        primary = self._get_primary()
        if primary is not None:
            try:
                primary.set(key, value)
            except Exception:
                self._disable_primary()
        self._fallback.set(key, value)

    def clear(self) -> None:
        primary = self._get_primary()
        if primary is not None:
            try:
                primary.clear()
            except Exception:
                self._disable_primary()
        self._fallback.clear()

    def _get_primary(self) -> CacheBackend[ValueT] | None:
        with self._lock:
            if not self._primary_available:
                return None
            return self._primary

    def _disable_primary(self) -> None:
        with self._lock:
            self._primary_available = False


class RedisCache(Generic[ValueT]):
    """Redis-backed TTL cache with explicit serialization hooks."""

    def __init__(
        self,
        *,
        redis_url: str,
        ttl_seconds: int = 600,
        serializer: Callable[[ValueT], str],
        deserializer: Callable[[str], ValueT],
        namespace: str = "mind_genrec",
    ) -> None:
        try:
            import redis
        except ImportError:
            redis = None

        self._client = (
            redis.Redis.from_url(redis_url, decode_responses=True)
            if redis is not None
            else _SimpleRedisClient(redis_url)
        )
        self._ttl_seconds = ttl_seconds
        self._serializer = serializer
        self._deserializer = deserializer
        self._namespace = namespace

    def get(self, key: str) -> ValueT | None:
        raw = self._client.get(self._full_key(key))
        if raw is None:
            return None
        return self._deserializer(raw)

    def set(self, key: str, value: ValueT) -> None:
        payload = self._serializer(value)
        self._client.set(self._full_key(key), payload, ex=self._ttl_seconds)

    def clear(self) -> None:
        pattern = f"{self._namespace}:*"
        cursor = 0
        while True:
            cursor, keys = self._client.scan(cursor=cursor, match=pattern, count=500)
            if keys:
                self._client.delete(*keys)
            if cursor == 0:
                break

    def _full_key(self, key: str) -> str:
        return f"{self._namespace}:{key}"


class _SimpleRedisClient:
    """Very small RESP client for environments without redis-py.

    This covers only the commands needed by the serving cache:
    `AUTH`, `SELECT`, `GET`, `SET EX`, `SCAN`, and `DEL`.
    """

    def __init__(self, redis_url: str) -> None:
        parsed = urlparse(redis_url)
        if parsed.scheme not in {"redis", "rediss"}:
            raise ValueError("Redis URL must use redis:// or rediss://")
        if not parsed.hostname or not parsed.port:
            raise ValueError("Redis URL must include host and port")

        self._host = parsed.hostname
        self._port = parsed.port
        self._use_ssl = parsed.scheme == "rediss"
        self._username = unquote(parsed.username) if parsed.username else None
        self._password = unquote(parsed.password) if parsed.password else None
        path = parsed.path.lstrip("/")
        self._db_index = int(path) if path else 0

    def get(self, key: str) -> str | None:
        return self._execute("GET", key)

    def set(self, key: str, value: str, *, ex: int) -> None:
        self._execute("SET", key, value, "EX", str(ex))

    def scan(self, *, cursor: int, match: str, count: int) -> tuple[int, list[str]]:
        raw = self._execute("SCAN", str(cursor), "MATCH", match, "COUNT", str(count))
        next_cursor = int(raw[0])
        keys = [str(key) for key in raw[1]]
        return next_cursor, keys

    def delete(self, *keys: str) -> int:
        if not keys:
            return 0
        return int(self._execute("DEL", *keys))

    def _execute(self, *parts: str):
        connection = socket.create_connection((self._host, self._port), timeout=5.0)
        if self._use_ssl:
            connection = ssl.create_default_context().wrap_socket(
                connection,
                server_hostname=self._host,
            )
        reader = connection.makefile("rb")
        try:
            if self._password is not None:
                if self._username is not None:
                    self._write_command(connection, "AUTH", self._username, self._password)
                else:
                    self._write_command(connection, "AUTH", self._password)
                self._read_response(reader)
            if self._db_index != 0:
                self._write_command(connection, "SELECT", str(self._db_index))
                self._read_response(reader)

            self._write_command(connection, *parts)
            return self._read_response(reader)
        finally:
            reader.close()
            connection.close()

    @staticmethod
    def _write_command(connection: socket.socket, *parts: str) -> None:
        encoded_parts = [part.encode("utf-8") for part in parts]
        chunks = [f"*{len(encoded_parts)}\r\n".encode("utf-8")]
        for part in encoded_parts:
            chunks.append(f"${len(part)}\r\n".encode("utf-8"))
            chunks.append(part + b"\r\n")
        connection.sendall(b"".join(chunks))

    def _read_response(self, reader):
        prefix = reader.read(1)
        if not prefix:
            raise RuntimeError("Redis connection closed unexpectedly")
        if prefix == b"+":
            return self._read_line(reader)
        if prefix == b"-":
            raise RuntimeError(f"Redis error: {self._read_line(reader)}")
        if prefix == b":":
            return int(self._read_line(reader))
        if prefix == b"$":
            length = int(self._read_line(reader))
            if length == -1:
                return None
            payload = reader.read(length)
            reader.read(2)
            return payload.decode("utf-8")
        if prefix == b"*":
            length = int(self._read_line(reader))
            if length == -1:
                return None
            return [self._read_response(reader) for _ in range(length)]
        raise RuntimeError(f"Unsupported Redis RESP prefix: {prefix!r}")

    @staticmethod
    def _read_line(reader) -> str:
        return reader.readline().rstrip(b"\r\n").decode("utf-8")
