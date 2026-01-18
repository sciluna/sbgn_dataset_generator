FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Set `DEVCONTAINER` environment variable to help with orientation
ENV DEVCONTAINER=true

# Set the default editor and visual
ENV EDITOR=nano
ENV VISUAL=nano

# Install basic development tools and iptables/ipset
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  less \
  sudo \
  man-db \
  ripgrep \
  fzf \
  unzip \
  gnupg2 \
  jq \
  python3-pip \
  python3-setuptools \
  python3-wheel \
  wget \
  curl \
  gh \
  git \
  nano \
  micro \
  ncdu \
  htop \
  screen \
  fontconfig \
  fonts-dejavu-core \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv (standalone Rust-based Python tool manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Node.js & Codex
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g @openai/codex@latest \
    && rm -rf /var/lib/apt/lists/*

# Create workspace and config directories and set permissions
RUN mkdir -p /workspace /root/.codex

WORKDIR /workspace
