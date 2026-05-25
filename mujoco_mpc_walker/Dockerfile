FROM python:3.10-slim

# Instalar dependências básicas do sistema necessárias pelo MuJoCo
RUN apt-get update && apt-get install -y \
    libglew-dev \
    libglfw3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar os arquivos do projeto
COPY . .

# Comando padrão
CMD ["python", "train.py"]
