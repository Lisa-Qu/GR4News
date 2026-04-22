FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY mind_genrec/requirements.txt /app/mind_genrec/requirements.txt
RUN pip install --no-cache-dir -r /app/mind_genrec/requirements.txt

COPY mind_genrec /app/mind_genrec

EXPOSE 8000

CMD ["python", "-m", "mind_genrec.serving.run_server"]
