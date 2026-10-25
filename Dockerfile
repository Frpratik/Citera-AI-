# One image for the web app AND the background worker (they differ only in the command).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/app/.cache/huggingface

# Do not run as root: if the app were ever compromised, it could do much less.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app

WORKDIR /app

# CPU-only PyTorch first. The default wheel bundles GPU (CUDA) libraries and adds gigabytes
# to the image; this app runs its small AI models on the CPU.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Dependencies before code, so editing the code does not re-install everything.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app app ./app
COPY --chown=app:app migrations ./migrations
COPY --chown=app:app frontend ./frontend

USER app
EXPOSE 8000

# "Is the web app answering?" (the worker container has no web server and overrides this).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
