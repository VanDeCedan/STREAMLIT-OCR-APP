# 1. Image de base
FROM python:3.12.9-slim

# 2. Empêcher Python de créer des fichiers .pyc et activer log instantané
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Installer dépendances système (ex: pour PyTorch, opencv)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Définir répertoire de travail
WORKDIR /app

# 5. Copier requirements et installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copier tout le code
COPY . .

# 7. Lancer l'app Streamlit
CMD ["sh", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]