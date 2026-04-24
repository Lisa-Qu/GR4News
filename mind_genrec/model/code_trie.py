"""Prefix tree (Trie) over valid semantic code sequences.

Used to constrain beam search so only existing item codes are generated.
"""

from __future__ import annotations

from typing import Iterator


class CodeTrie:
    """Trie built from all valid semantic code sequences.

    Each node stores the set of valid next tokens. Leaf nodes (at depth
    == code_length) map back to item IDs.
    """

    def __init__(self) -> None:
        # children[token] -> child CodeTrie node
        self._children: dict[int, "CodeTrie"] = {}
        self._items: list[str] = []

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def insert(self, code: tuple[int, ...] | list[int], items: list[str]) -> None:
        """Insert one code sequence and its associated item IDs."""
        node = self
        for token in code:
            if token not in node._children:
                node._children[token] = CodeTrie()
            node = node._children[token]
        node._items = list(items)

    @classmethod
    def from_code_to_items(
        cls, code_to_items: dict[tuple[int, ...], list[str]]
    ) -> "CodeTrie":
        trie = cls()
        for code, items in code_to_items.items():
            trie.insert(code, items)
        return trie

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def valid_next_tokens(self, prefix: tuple[int, ...] | list[int]) -> list[int]:
        """Return all valid next tokens given a prefix (excluding BOS token)."""
        node = self._find_node(prefix)
        if node is None:
            return []
        return list(node._children.keys())

    def items_at(self, code: tuple[int, ...] | list[int]) -> list[str]:
        """Return item IDs for a complete code sequence."""
        node = self._find_node(code)
        if node is None:
            return []
        return list(node._items)

    def is_valid_prefix(self, prefix: tuple[int, ...] | list[int]) -> bool:
        return self._find_node(prefix) is not None

    def all_codes(self) -> Iterator[tuple[int, ...]]:
        """Iterate over all complete code sequences in the trie."""
        yield from self._iter_codes(())

    def __len__(self) -> int:
        return sum(1 for _ in self.all_codes())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_node(
        self, prefix: tuple[int, ...] | list[int]
    ) -> "CodeTrie | None":
        node: CodeTrie = self
        for token in prefix:
            child = node._children.get(token)
            if child is None:
                return None
            node = child
        return node

    def _iter_codes(self, prefix: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
        if self._items:
            yield prefix
        for token, child in self._children.items():
            yield from child._iter_codes(prefix + (token,))
