# SPANIEL demo app — free-form PII extraction, constrained decoding, local-only.
#
#   docker run -p 8377:8377 ghcr.io/harshsinghal/spaniel
#
# The model (~1.2GB) is pulled from the Hugging Face hub on first start and
# cached in the /models volume; mount it to persist across runs:
#   docker run -p 8377:8377 -v spaniel-models:/models ghcr.io/harshsinghal/spaniel
# On Linux with an NVIDIA GPU, add: --gpus all

FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    transformers accelerate fastapi uvicorn huggingface_hub

COPY pii_decode.py /app/
COPY pii_tagger/server.py /app/pii_tagger/
COPY pii_tagger/static/ /app/pii_tagger/static/

ENV PII_MODEL_ID=Harsh/qwen3-0.6b-pii-sft-v2 \
    PII_MODEL_CACHE=/models \
    HF_HUB_DISABLE_XET=1

VOLUME /models
EXPOSE 8377

# server.py does `sys.path.insert(parent)` to import pii_decode — /app is the parent
CMD ["python", "-m", "uvicorn", "server:app", "--app-dir", "/app/pii_tagger", \
     "--host", "0.0.0.0", "--port", "8377"]
