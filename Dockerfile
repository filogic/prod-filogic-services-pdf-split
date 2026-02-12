FROM python:3.12-slim

# Install Tesseract OCR + language packs + Poppler (pdf2image dependency)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-nld \
    tesseract-ocr-eng \
    tesseract-ocr-deu \
    tesseract-ocr-fra \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Cloud Run uses PORT env var
ENV PORT=8080

CMD exec functions-framework --target=split_pdf_handler --port=${PORT}
