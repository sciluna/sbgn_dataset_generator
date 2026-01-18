#!/usr/bin/env bash
set -euo pipefail

# Install system packages required by Playwright's Chromium runtime.
if command -v apt-get >/dev/null 2>&1; then
  apt-get update
  apt-get install -y \
    libglib2.0-0t64 \
    libnspr4 \
    libnss3 \
    libdbus-1-3 \
    libatk1.0-0t64 \
    libatk-bridge2.0-0t64 \
    libcups2t64 \
    libxcb1 \
    libxkbcommon0 \
    libatspi2.0-0t64 \
    libx11-6 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libcairo2 \
    libpango-1.0-0 \
    libasound2t64
else
  echo "apt-get not found; install Playwright system dependencies manually." >&2
  exit 1
fi

# Fetch the Chromium browser binaries expected by Playwright.
uv run --with playwright -m playwright install chromium
