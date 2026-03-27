FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    openssh-server \
    curl \
    ca-certificates \
    rsync \
    jq \
    tmux \
    pigz \
    unzip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages setuptools wheel

# Pinned competition/runtime stack for our runs. We install zstandard
# explicitly because missing it caused silent fallback to zlib on expensive pods.
RUN python3 -m pip install --break-system-packages \
    torch==2.10 \
    numpy \
    huggingface-hub \
    sentencepiece \
    zstandard

WORKDIR /workspace

# These helpers live in the image so a fresh pod can validate itself before we
# sync any local repo state onto it.
COPY ops/runpod_image/validate_image.sh /opt/parameter-golf/validate_image.sh
COPY ops/runpod_image/preflight_report.py /opt/parameter-golf/preflight_report.py
COPY ops/runpod_image/start_container.sh /opt/parameter-golf/start_container.sh

RUN chmod +x /opt/parameter-golf/validate_image.sh /opt/parameter-golf/start_container.sh

# Start sshd, install PUBLIC_KEY if present, and keep the container alive so
# RunPod direct TCP SSH/SCP/rsync work normally.
CMD ["bash", "/opt/parameter-golf/start_container.sh"]
