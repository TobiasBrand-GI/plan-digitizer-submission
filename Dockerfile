# Same GDAL 3.10.0 / Python 3.12 base as the dev container (.devcontainer/gdal)
# -- GDAL's Python bindings (osgeo) come prebuilt with it, so the container
# runs exactly the combination the code is developed and tested against.
FROM windpioneers/gdal-python:boiling-gopher-gdal-3.10.0-python-3.12-dev

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ffmpeg, libsm6, libxext6: runtime libraries of opencv-python (non-headless)
# tesseract-ocr: the OCR engine behind pytesseract
RUN apt-get update \
 && apt-get install -y ffmpeg libsm6 libxext6 tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements first, so code changes don't invalidate the pip layer
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Non-root user (reuses the image's vscode user if it exists) owning runs/,
# where every job writes its uploads and results
RUN (id -u vscode >/dev/null 2>&1 || useradd --create-home --uid 1000 vscode) \
 && mkdir -p runs \
 && chown -R vscode:vscode runs
USER vscode

EXPOSE 8000

# One worker on purpose: the web UI's job list lives in this process's memory
CMD ["fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]
