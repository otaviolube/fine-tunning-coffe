FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Configurar timezone não interativo
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Sao_Paulo

# Instalar dependências básicas
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Criar diretório de trabalho
WORKDIR /app

# Instalar PyTorch (versão específica com CUDA)
RUN pip3 install --no-cache-dir torch==2.1.0+cu121 torchvision==0.16.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# Clonar repositório LLaVA
RUN git clone https://github.com/haotian-liu/LLaVA.git

# Instalar dependências do Python
RUN cd LLaVA && \
    pip3 install --upgrade pip && \
    pip3 install -e . && \
    pip3 install -e ".[train]" && \
    pip3 install flash-attn --no-build-isolation && \
    pip3 install deepspeed && \
    pip3 install wandb && \
    pip3 install datasets && \
    pip3 install --upgrade --force-reinstall Pillow && \
    pip3 install bitsandbytes==0.41.1

# Configurar variáveis de ambiente CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Script para executar o treinamento
COPY train.sh /app/train.sh
RUN chmod +x /app/train.sh

# Volume para dados e checkpoints
VOLUME ["/app/dataset", "/app/checkpoints"]

# Comando padrão
CMD ["/bin/bash"]