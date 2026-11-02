# Hugging Face Space (Docker SDK) for the PTIC research demo. CPU hardware, port 7860.
#
# The model weights, the input-check statistics and the sample images are NOT in this repository: they live in a private
# Hugging Face repo (see fetch_assets.py) and are downloaded with the Space secret HF_TOKEN, at build time when the secret is
# available to the build and otherwise when the container starts.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    PTIC_DEMO_PUBLIC=1 PTIC_DEMO_DEVICE=cpu PTIC_DEMO_MAX_MB=25 PTIC_DEMO_MAX_MPIX=40 \
    PTIC_DEMO_FRAME_ANCESTORS="https://huggingface.co https://*.hf.space"

# Libraries as root (so that `python -I`, which ignores the user site, still finds them), the app as the unprivileged user 1000
# that Spaces run containers as.
WORKDIR /tmp/build
COPY requirements.txt .
RUN pip install -r requirements.txt && rm -rf /tmp/build
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
COPY --chown=user . .

# Build-time download of the private assets: faster cold starts. Not fatal if the secret is not offered to the build; start-up retries.
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=false \
    python fetch_assets.py || echo "assets not fetched at build time; they are fetched when the container starts"

EXPOSE 7860
CMD ["sh", "-c", "python fetch_assets.py && exec python -I -m uvicorn src.demo.app:app --app-dir /home/user/app --host 0.0.0.0 --port 7860 --workers 1 --no-server-header"]
