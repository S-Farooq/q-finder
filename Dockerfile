FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py build_index.py hafs_smart_v8.json .
COPY quran_tracker quran_tracker

# Pre-build index so startup is fast
RUN python build_index.py

EXPOSE 8000

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
