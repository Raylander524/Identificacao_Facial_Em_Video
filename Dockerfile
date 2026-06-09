# =========================
# Stage 1 - Builder
# =========================
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    wget \
    unzip \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    git \
    ffmpeg && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# Instala dependências restantes
RUN pip install --no-cache-dir -r requirements.txt

# Download do modelo InsightFace
RUN mkdir -p /root/.insightface/models/buffalo_l && \
    wget -O /root/.insightface/models/buffalo_l.zip \
    https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && \
    unzip /root/.insightface/models/buffalo_l.zip \
    -d /root/.insightface/models/buffalo_l/ && \
    rm /root/.insightface/models/buffalo_l.zip

COPY . .


# =========================
# Stage 2 - Runtime
# =========================
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/bin:$PATH"

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    ffmpeg && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Copia ambiente Python do builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copia modelos InsightFace
COPY --from=builder /root/.insightface /root/.insightface

# Copia aplicação
COPY --from=builder /app /app

EXPOSE 5000

CMD ["python", "main.py"]