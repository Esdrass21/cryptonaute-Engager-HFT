# Cryptonaute HFT — production/runtime image.
# Multi-stage; the same image runs the platform, the validation harness, and (optionally)
# the Prometheus/Grafana sidecars via docker-compose.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

# System deps for numpy/scipy/TA-Lib/polars wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates libta-lib-dev wget \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, lockfile-reproducible installs.
RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml ./
# Install core deps first (layer cache), then the package.
RUN uv pip install --system \
    "numpy>=1.26,<3" "pandas>=2.2,<3" "polars>=1.0,<2" "scipy>=1.13,<2" \
    "statsmodels>=0.14,<1" "scikit-learn>=1.5,<2" "pandas-ta>=0.3.14b0,<1" \
    "transformers>=4.40,<5" "torch>=2.2,<3" "feedparser>=6.0,<7" "httpx>=0.28.1,<1" \
    "beautifulsoup4>=4.12,<5" "MetaTrader5>=5.0.45,<6" "ccxt>=4.3,<5" \
    "apscheduler>=3.10,<4" "pyyaml>=6.0,<7" "structlog>=24.1,<25" "pydantic>=2.7,<3" \
    "python-dotenv>=1.0,<2" "transitions>=0.9,<1" "redis>=5.0,<6" "psycopg[binary]>=3.1,<4" \
    "prometheus-client>=0.20,<1" "arch>=7.0,<8" \
    && uv pip install --system "uvicorn>=0.30,<1"

COPY . .
RUN uv pip install --system -e . --no-deps

# Non-root for least-privilege (architecture §8: least-privilege).
RUN useradd -m -u 1000 cryptonaute && chown -R cryptonaute:cryptonaute /app
USER cryptonaute

ENTRYPOINT ["python", "-m", "cryptonaute_hft.main"]
CMD ["--paper", "--steps", "0"]
