# SPANIEL demo app — free-form PII extraction, constrained decoding, local-only.
#
#   docker run -p 8377:8377 ghcr.io/harshsinghal/spaniel
#
# The model (~1.2GB) is pulled from the Hugging Face hub on first start and
# cached in the /models volume; mount it to persist across runs:
#   docker run -p 8377:8377 -v spaniel-models:/models ghcr.io/harshsinghal/spaniel
# This published image intentionally uses CPU PyTorch for portability.

FROM python:3.12-slim-bookworm

WORKDIR /app

ARG TORCH_VERSION=2.8.0
RUN pip install --no-cache-dir \
    "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu

COPY requirements-demo.txt /app/
RUN pip install --no-cache-dir -r requirements-demo.txt

COPY pii_decode.py /app/
COPY tokenizer_compat.py /app/
COPY model_compat.py /app/
COPY pii_tagger/server.py /app/pii_tagger/
COPY pii_tagger/static/ /app/pii_tagger/static/

ENV PII_MODEL_ID=Harsh/qwen3-0.6b-pii-sft-v2 \
    PII_MODEL_REVISION=fc31a21a2ac54850ba3c12781ae433413691189d \
    PII_MODEL_CACHE=/models \
    HF_HUB_DISABLE_XET=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 10001 spaniel \
    && mkdir -p /models \
    && chown spaniel:spaniel /models
VOLUME /models
EXPOSE 8377
USER spaniel

HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8377/', timeout=4)"

# server.py does `sys.path.insert(parent)` to import pii_decode — /app is the parent
CMD ["python", "-m", "uvicorn", "server:app", "--app-dir", "/app/pii_tagger", \
     "--host", "0.0.0.0", "--port", "8377"]
