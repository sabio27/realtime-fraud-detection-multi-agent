# Image Linux légère avec Python 3.11
FROM python:3.11-slim

# Dossier de travail dans le container
WORKDIR /app

# Copie les dépendances d'abord (cache Docker)
COPY requirements.txt .

# Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copie tous les fichiers du projet
COPY . .

# Port Streamlit
EXPOSE 8501

# Lancement
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]