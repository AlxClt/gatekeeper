# syntax=docker/dockerfile:1

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install the library (and API extras) first for better caching
COPY pyproject.toml README.md /app/
COPY src/ /app/src/
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir ".[api]"

# Then copy the API wrapper
COPY api/ /app/api/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

