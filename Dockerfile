FROM python:3.10-slim

WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Copier les fichiers nécessaires
COPY pyproject.toml /app/
COPY src /app/src
COPY artifacts /app/artifacts
COPY data /app/data

# Installer Hatch et créer l'environnement
RUN pip install --no-cache-dir hatch \
    && hatch env create

ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ENABLEXSRSFPROTECTION=false \
    STREAMLIT_SERVER_ALLOW_ORIGIN="*"

EXPOSE 8501

CMD ["sh", "-c", "PORT=${PORT:-8501} hatch run webapp --server.port=$PORT"]
