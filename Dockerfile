FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY agent_memory_manager ./agent_memory_manager

RUN pip install --no-cache-dir ".[all]" fastapi uvicorn

# Default server entrypoint — override via CMD or docker-compose
CMD ["uvicorn", "agent_memory_manager.server.entrypoint:app", "--host", "0.0.0.0", "--port", "8000"]
