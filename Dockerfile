FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and assets
COPY backend/ backend/
COPY data/spotify_tracks_modeled.csv data/spotify_tracks_modeled.csv
COPY models/ models/
COPY frontend/ frontend/

EXPOSE 8000

CMD exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
