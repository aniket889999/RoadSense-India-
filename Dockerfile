# RoadSense ANPR API (CPU) — local/academic container. Models and data are NOT baked in:
# mount ./models read-only and ./outputs read-write (see infra/docker-compose.yml, profile "api").
# Docker on macOS cannot use Apple MPS, so the plate detector runs on CPU in this image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_AUTOINSTALL=false \
    MPLBACKEND=Agg

RUN apt-get update \
 && apt-get install --yes --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-anpr.txt requirements.txt ./
COPY services/api/requirements.txt services/api/requirements.txt
# Same pinned set as CI (.github/workflows/ci.yml); CPU wheels for torch/torchvision.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 torchvision==0.28.0 \
 && pip install -r requirements-anpr.txt -r requirements.txt -r services/api/requirements.txt \
      httpx==0.28.1 python-multipart==0.0.32 greenlet==3.5.5 asyncpg==0.31.0 lap==0.5.13 \
      PyYAML==6.0.3 Pillow==12.3.0 pandas==3.0.5

COPY src ./src
COPY services ./services
COPY scripts ./scripts
COPY configs ./configs
COPY data/reference ./data/reference

RUN useradd --create-home --uid 10001 roadsense \
 && mkdir -p /app/outputs /app/models \
 && chown -R roadsense:roadsense /app/outputs
USER roadsense

# Selected RapidOCR review profile, mirroring scripts/serve_anpr_local.py, with a CPU detector.
ENV ROADSENSE_ANPR_PLATE_WEIGHTS=models/anpr/anpr_witvix_detector_mps_v1_best.pt \
    ROADSENSE_ANPR_PLATE_SHA256=37eae626c753e3742bc27437f6c7671898315a4591e2d0cf92e98d996b0421db \
    ROADSENSE_ANPR_OCR_ENGINE=rapidocr \
    ROADSENSE_ANPR_OCR_MODEL_DIR=models/rapidocr \
    ROADSENSE_ANPR_OCR_CROP_VIEW=resized_bgr \
    ROADSENSE_ANPR_DEVICE=cpu \
    DATABASE_URL=sqlite+aiosqlite:///outputs/roadsense.db

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "services.api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
