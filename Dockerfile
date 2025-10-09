# Stage 1: Build (com ferramentas de compilação)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev build-essential \
    wget unzip ca-certificates libgl1-mesa-glx libglib2.0-0 git \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

COPY requirements.txt .

# Instala todas as dependências Python
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /tmp/*


# Baixa e prepara o modelo buffalo_l
RUN mkdir -p /root/.insightface/models/buffalo_l && \
    wget -O /root/.insightface/models/buffalo_l.zip https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && \
    unzip /root/.insightface/models/buffalo_l.zip -d /root/.insightface/models/buffalo_l/ && \
    rm /root/.insightface/models/buffalo_l.zip

# Copia código da aplicação
COPY . .

# Stage 2: Imagem final leve
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/bin:$PATH"

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip libgl1-mesa-glx libglib2.0-0 ca-certificates wget unzip git \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Copia bibliotecas Python do builder
COPY --from=builder /usr/lib/python3/dist-packages /usr/lib/python3/dist-packages
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copia modelos e código da aplicação
COPY --from=builder /root/.insightface /root/.insightface
COPY --from=builder /app /app

COPY cert.pem key.pem /app/

EXPOSE 5000

CMD ["python", "main.py"]
