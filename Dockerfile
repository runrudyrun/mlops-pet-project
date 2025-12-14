FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends git \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY configs /app/configs
COPY .dvc /app/.dvc
COPY .dvcignore /app/.dvcignore
COPY dvc.yaml /app/dvc.yaml
COPY dvc.lock /app/dvc.lock
COPY models /app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; port=os.environ.get('PORT','8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health').read(); print('ok')" || exit 1

CMD ["sh", "-c", "mkdir -p models data/processed && \
  dvc config core.no_scm true && \
  if [ -n \"${DAGSHUB_USERNAME:-}\" ] && [ -n \"${DAGSHUB_TOKEN:-}\" ]; then \
    dvc remote modify --local dagshub user \"$DAGSHUB_USERNAME\" && \
    dvc remote modify --local dagshub password \"$DAGSHUB_TOKEN\"; \
  else \
    echo 'ERROR: DAGSHUB_USERNAME and DAGSHUB_TOKEN must be set to pull DVC data from DagsHub.' 1>&2; \
    exit 1; \
  fi && \
  dvc pull models/model.pkl models/model_info.txt data/processed/train.csv data/processed/test.csv && \
  uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
