FROM python:3.11-slim

# System dependencies for MuJoCo physics + headless (OSMesa/EGL) rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglx-mesa0 \
    libgl1-mesa-dev \
    libglew-dev \
    libosmesa6-dev \
    libglfw3-dev \
    libegl-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=osmesa

WORKDIR /app

# Install Python deps first (better layer caching).
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the project (.dockerignore excludes .git/.venv/runs/checkpoints).
COPY . .

CMD ["bash"]
