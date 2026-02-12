"""
Cloud Function: PDF Splitter by Reference
Splits a PDF into multiple documents based on a reference pattern (e.g. order number)
found in the text of each page. Supports both text-based and scanned (image) PDFs
via OCR fallback (Tesseract).

Deploy:
  export GCP_PROJECT_ID=my-project
  ./deploy.sh
"""
from __future__ import annotations

import functions_framework
import io
import json
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor

import pdfplumber
from pypdf import PdfReader, PdfWriter
from google.cloud import storage

# OCR imports — optional, only needed for scanned PDFs
try:
    import shutil
    import pytesseract
    from pdf2image import convert_from_bytes
    # Lightweight check: just verify the binaries exist on PATH
    # (avoids slow convert_from_bytes probe on every cold start)
    OCR_AVAILABLE = bool(
        shutil.which("tesseract") and shutil.which("pdftoppm")
    )
except ImportError:
    OCR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GCS_BUCKET = os.environ.get("GCS_BUCKET", "pdf-splitter-output")
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "20"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
OCR_DPI = int(os.environ.get("OCR_DPI", "120"))
OCR_LANGUAGES = os.environ.get("OCR_LANGUAGES", "nld+eng")
OCR_CROP_TOP_PCT = float(os.environ.get("OCR_CROP_TOP_PCT", "0.25"))
OCR_WORKERS = int(os.environ.get("OCR_WORKERS", "4"))
# Minimum text length from cropped OCR to consider the page "readable".
# Pages below this threshold get orientation detection + full-page retry.
OCR_MIN_TEXT_LEN = int(os.environ.get("OCR_MIN_TEXT_LEN", "20"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ocr_one_page(img):
    """OCR a single page image: crop to top portion, return (text, img).

    No orientation detection here — that is deferred to the retry pass
    for pages that produce very little text (keeps the fast path fast).
    """
    if OCR_CROP_TOP_PCT < 1.0:
        w, h = img.size
        cropped = img.crop((0, 0, w, int(h * OCR_CROP_TOP_PCT)))
    else:
        cropped = img
    text = pytesseract.image_to_string(cropped, lang=OCR_LANGUAGES)
    return text, img


def _ocr_retry_page(args):
    """Full-page OCR retry for an unmatched page.

    Accepts (img, cropped_text) where cropped_text is the text from the
    first pass (cropped top 25%).  Strategy:
      1. Run full-page OCR at original orientation.
      2. If the page had very little or garbled text in the first pass
         (< 200 chars), also try 90°, 180°, 270° rotations and pick
         the orientation with the most text.  This handles rotated pages.
      3. Pages with substantial first-pass text (>= 200 chars) are NOT
         rotated — they just had the reference below the crop line, so
         a single full-page OCR at 0° is enough.
    """
    img, cropped_text = args

    best_text = pytesseract.image_to_string(img, lang=OCR_LANGUAGES)
    best_img = img

    # Only try rotations if the first pass produced very little text,
    # indicating the page is likely rotated (garbled OCR).
    if len(cropped_text.strip()) < 200:
        for test_angle in [90, 180, 270]:
            rotated = img.rotate(-test_angle, expand=True)
            text = pytesseract.image_to_string(rotated, lang=OCR_LANGUAGES)
            if len(text.strip()) > len(best_text.strip()):
                best_text = text
                best_img = rotated

    return best_text, best_img


def extract_text_per_page(
    pdf_bytes: bytes,
    ocr_mode: str = "auto",
) -> tuple[list[str], str, list | None]:
    """
    Extract text from each page.

    ocr_mode:
      - "auto": Try pdfplumber first; if ALL pages are empty, fall back to OCR.
      - "force": Always use OCR (for scanned documents).
      - "off": Only use pdfplumber (no OCR).

    Returns (page_texts, method_used, corrected_images) where:
      - method_used is "text" or "ocr"
      - corrected_images is a list of orientation-corrected PIL images
        when OCR was used (None otherwise).  These can be re-used for
        full-page OCR retries on unmatched pages.
    """
    # ── Try text extraction first (unless force OCR) ──────────
    if ocr_mode != "force":
        texts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                texts.append(text)

        # If we got meaningful text, use it
        has_text = any(len(t.strip()) > 10 for t in texts)
        if has_text or ocr_mode == "off":
            return texts, "text", None

    # ── Fall back to OCR ──────────────────────────────────────
    if not OCR_AVAILABLE:
        if ocr_mode == "force":
            raise RuntimeError(
                "OCR is not available. Install poppler-utils and pytesseract, "
                "or deploy with Docker (see Dockerfile)."
            )
        # auto mode: return empty texts with a warning — the caller will
        # report NO_REFERENCES_FOUND with a helpful hint
        return texts, "text", None

    try:
        images = convert_from_bytes(pdf_bytes, dpi=OCR_DPI)
    except Exception as e:
        if ocr_mode == "force":
            raise RuntimeError(f"OCR failed: {e}. Is poppler-utils installed?")
        # auto mode: return empty texts
        return texts, "text", None

    # ── First pass: parallel cropped OCR (no OSD) ─────────────
    with ThreadPoolExecutor(max_workers=OCR_WORKERS) as pool:
        results = list(pool.map(_ocr_one_page, images))

    ocr_texts = [r[0] for r in results]
    page_images = [r[1] for r in results]  # originals (not rotated yet)

    return ocr_texts, "ocr", page_images


def find_references(
    page_texts: list[str],
    pattern: str,
    search_area: str = "any",
) -> list[dict]:
    """
    Find the *first* reference on each page.
    Returns a list of {page_index, reference} dicts — one per page (or
    None for pages with no match).
    """
    compiled = re.compile(pattern, re.IGNORECASE)
    matches = []

    for page_idx, text in enumerate(page_texts):
        search_text = text
        if search_area == "first_line":
            search_text = text.split("\n")[0] if text else ""
        elif search_area == "header":
            lines = text.split("\n")
            search_text = "\n".join(lines[:5]) if lines else ""

        match = compiled.search(search_text)
        if match:
            # Use first capture group if available, otherwise full match.
            # Strip internal whitespace (OCR sometimes splits digits).
            ref = match.group(1) if match.lastindex and match.lastindex >= 1 else match.group()
            ref = re.sub(r'\s+', '', ref)
            matches.append({"page_index": page_idx, "reference": ref})
        else:
            matches.append(None)

    return matches


def group_pages_by_reference(
    per_page_matches: list[dict | None],
    total_pages: int,
    split_mode: str = "before_reference",
) -> tuple[dict[str, list[int]], list[int]]:
    """
    Group pages into documents based on where references are found.

    Each page is assigned to the reference it carries. Pages without a
    match are assigned to the *current* reference only as long as no new
    (different) reference appears later — i.e. they are "continuation"
    pages.  However, trailing pages at the end of the document that carry
    no reference, or pages sandwiched between two *different* references,
    are collected as unmatched.

    split_mode:
      - 'before_reference': A new document starts AT the page where a
        *new* reference is found (most common: reference is on the first
        page of each order).
      - 'after_reference': A new document starts AFTER the page where a
        *new* reference is found (reference is on a separator/cover page).

    Returns:
      - documents: {reference: [page_numbers (1-based)]}
      - unmatched_pages: [page_numbers (1-based)] for pages that could
        not be attributed to any reference
    """
    if not any(m is not None for m in per_page_matches):
        return {}, list(range(1, total_pages + 1))

    # --- Build a per-page reference map ---
    # For each page, determine which reference it belongs to.
    # A page with an explicit match gets that reference.
    # A page without a match is tentatively assigned to the preceding
    # reference, but only if the *next* page with a match carries the
    # same reference (i.e. it's a true continuation page).  Otherwise
    # the page is unmatched.

    # First, build a list of (page_index, reference) for pages that have
    # an explicit match.
    explicit: list[tuple[int, str]] = []
    for entry in per_page_matches:
        if entry is not None:
            explicit.append((entry["page_index"], entry["reference"]))

    # For every page, look up the *next* explicit reference at or after
    # that page, and the *previous* explicit reference at or before it.
    # Pre-compute prev_ref and next_ref arrays for O(n) lookup.
    prev_ref: list[str | None] = [None] * total_pages
    next_ref: list[str | None] = [None] * total_pages

    last = None
    for i in range(total_pages):
        if per_page_matches[i] is not None:
            last = per_page_matches[i]["reference"]
        prev_ref[i] = last

    last = None
    for i in range(total_pages - 1, -1, -1):
        if per_page_matches[i] is not None:
            last = per_page_matches[i]["reference"]
        next_ref[i] = last

    # Now assign each page.
    page_assignment: list[str | None] = [None] * total_pages

    for i in range(total_pages):
        match = per_page_matches[i]
        if match is not None:
            # Page has an explicit reference
            page_assignment[i] = match["reference"]
        else:
            # No match — assign to the preceding reference only if the
            # next explicit reference is the same (true continuation).
            # Trailing pages (where next_ref is None) are NOT absorbed;
            # they become unmatched / "unknown".
            pr = prev_ref[i]
            nr = next_ref[i]
            if pr is not None and nr == pr:
                page_assignment[i] = pr
            # else: stays None → unmatched

    # --- Collect into documents and unmatched ---
    documents: dict[str, list[int]] = {}
    unmatched_pages: list[int] = []
    seen_refs: dict[str, int] = {}

    for i in range(total_pages):
        ref = page_assignment[i]
        if ref is None:
            unmatched_pages.append(i + 1)  # 1-based
        else:
            # Handle duplicate reference names (e.g. same order number
            # appearing in two non-contiguous groups).
            if ref not in seen_refs:
                seen_refs[ref] = 1
                documents[ref] = []
            documents[ref].append(i + 1)  # 1-based

    return documents, unmatched_pages


def merge_pdfs(pdf_list: list[bytes]) -> bytes:
    """Merge multiple PDFs into a single PDF."""
    writer = PdfWriter()
    for pdf_bytes in pdf_list:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            writer.add_page(page)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def split_pdf(pdf_bytes: bytes, page_groups: dict[str, list[int]]) -> dict[str, bytes]:
    """Split a PDF into multiple PDFs based on page groups."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    result = {}

    for reference, pages in page_groups.items():
        if not pages:
            continue
        writer = PdfWriter()
        for page_num in pages:
            writer.add_page(reader.pages[page_num - 1])  # Convert to 0-based

        buffer = io.BytesIO()
        writer.write(buffer)
        result[reference] = buffer.getvalue()

    return result


def upload_to_gcs(
    split_pdfs: dict[str, bytes],
) -> list[dict]:
    """Upload split PDFs to GCS as .tmp files, return document list."""
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    results = []

    for reference, pdf_bytes in split_pdfs.items():
        blob_name = f"{uuid.uuid4().hex}.tmp"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(pdf_bytes, content_type="application/pdf")

        safe_ref = re.sub(r'[^\w\-.]', '_', reference)
        results.append({
            "reference": reference,
            "filename": f"{safe_ref}.pdf",
            "gcs_path": f"gs://{GCS_BUCKET}/{blob_name}",
            "pages": [],          # filled in by caller
            "page_count": 0,      # filled in by caller
            "file_size_bytes": len(pdf_bytes),
        })

    return results


def make_error(message: str, error_code: str, status: int = 400, **extra):
    """Create a JSON error response."""
    body = {"status": "error", "error_code": error_code, "message": message, **extra}
    return (json.dumps(body, ensure_ascii=False), status, {"Content-Type": "application/json"})


def make_success(data, status: int = 200):
    """Create a JSON success response."""
    return (json.dumps(data, ensure_ascii=False), status, {"Content-Type": "application/json"})


# ---------------------------------------------------------------------------
# Cloud Function entry point
# ---------------------------------------------------------------------------
@functions_framework.http
def split_pdf_handler(request):
    """
    HTTP Cloud Function.

    Accepts multipart/form-data or application/json.

    Multipart fields:
      - file (or file[] or files): one or more PDF binaries
      - reference_pattern: regex pattern (required)
      - split_mode: "before_reference" | "after_reference" (default: before_reference)
      - search_area: "any" | "first_line" | "header" (default: any)

    JSON body:
      - file_base64: base64-encoded PDF (string or list of strings)
      - filename: original filename
      - reference_pattern: regex pattern (required)
      - split_mode / search_area: same as above

    When multiple files are provided they are merged into a single PDF
    before splitting by reference.
    """

    # ---- CORS handling ----
    if request.method == "OPTIONS":
        return ("", 204, {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        })

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Content-Type": "application/json",
    }

    if request.method != "POST":
        return make_error("Only POST is allowed", "METHOD_NOT_ALLOWED", 405)

    # ---- Parse input ----
    content_type = request.content_type or ""
    pdf_bytes = None
    filename = "unknown.pdf"
    reference_pattern = None
    split_mode = "before_reference"
    search_area = "any"
    ocr_mode = "auto"

    try:
        if "multipart/form-data" in content_type:
            # Accept any file field name — try known names first,
            # then fall back to the first file field in the request.
            files = (
                request.files.getlist("file")
                or request.files.getlist("file[]")
                or request.files.getlist("files")
            )
            if not files:
                # Fall back: grab files from whatever field name was used
                for key in request.files:
                    files = request.files.getlist(key)
                    if files:
                        break
            if not files:
                return make_error(
                    "Missing file in form data. "
                    f"Received fields: {list(request.form.keys())}, "
                    f"file fields: {list(request.files.keys())}",
                    "MISSING_FILE",
                )
            if len(files) == 1:
                pdf_bytes = files[0].read()
                filename = files[0].filename or filename
            else:
                # Multiple files — merge into a single PDF
                pdf_list = [f.read() for f in files]
                pdf_bytes = merge_pdfs(pdf_list)
                filename = "merged.pdf"
            reference_pattern = request.form.get("reference_pattern")
            split_mode = request.form.get("split_mode", split_mode)
            search_area = request.form.get("search_area", search_area)
            ocr_mode = request.form.get("ocr", ocr_mode)

        elif "application/json" in content_type:
            import base64
            body = request.get_json(silent=True) or {}
            file_b64 = body.get("file_base64")
            if not file_b64:
                return make_error("Missing 'file_base64' in JSON body", "MISSING_FILE")
            if isinstance(file_b64, list):
                # Multiple files — merge into a single PDF
                pdf_list = [base64.b64decode(b) for b in file_b64]
                pdf_bytes = merge_pdfs(pdf_list)
                filename = "merged.pdf"
            else:
                pdf_bytes = base64.b64decode(file_b64)
            filename = body.get("filename", filename)
            reference_pattern = body.get("reference_pattern")
            split_mode = body.get("split_mode", split_mode)
            search_area = body.get("search_area", search_area)
            ocr_mode = body.get("ocr", ocr_mode)

        else:
            return make_error(
                "Content-Type must be multipart/form-data or application/json",
                "INVALID_CONTENT_TYPE",
                415,
            )
    except Exception as e:
        return make_error(f"Failed to parse request: {e}", "PARSE_ERROR")

    # ---- Validate ----
    if not reference_pattern:
        return make_error("'reference_pattern' is required", "MISSING_PATTERN")

    try:
        re.compile(reference_pattern)
    except re.error as e:
        return make_error(f"Invalid regex pattern: {e}", "INVALID_PATTERN")

    if split_mode not in ("before_reference", "after_reference"):
        return make_error(
            "split_mode must be 'before_reference' or 'after_reference'",
            "INVALID_SPLIT_MODE",
        )

    if search_area not in ("any", "first_line", "header"):
        return make_error(
            "search_area must be 'any', 'first_line', or 'header'",
            "INVALID_SEARCH_AREA",
        )

    if ocr_mode not in ("auto", "force", "off"):
        return make_error(
            "ocr must be 'auto', 'force', or 'off'",
            "INVALID_OCR_MODE",
        )

    if ocr_mode == "force" and not OCR_AVAILABLE:
        return make_error(
            "OCR is not available. Deploy with Docker for Tesseract support, "
            "or use ocr='auto' to try text extraction first.",
            "OCR_NOT_AVAILABLE",
        )

    file_size_mb = len(pdf_bytes) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return make_error(
            f"File too large: {file_size_mb:.1f}MB (max {MAX_FILE_SIZE_MB}MB)",
            "FILE_TOO_LARGE",
        )

    # ---- Process ----
    try:
        # 1. Extract text from each page (with OCR fallback)
        page_texts, extraction_method, corrected_images = extract_text_per_page(
            pdf_bytes, ocr_mode
        )
        total_pages = len(page_texts)

        if total_pages == 0:
            return make_error("PDF has no pages", "EMPTY_PDF")

        # 2. Find references per page
        per_page_matches = find_references(page_texts, reference_pattern, search_area)

        # 2b. For OCR pages with no match, retry with full-page OCR.
        #     The first pass only scanned a cropped top portion — pages
        #     with the reference below the crop or rotated pages will
        #     have been missed.  We retry ALL unmatched pages; the retry
        #     function decides per-page whether to also try rotations
        #     based on how much text the first pass produced.
        if corrected_images:
            retry_indices = [
                idx for idx, match in enumerate(per_page_matches)
                if match is None and idx < len(corrected_images)
            ]
            if retry_indices:
                retry_args = [
                    (corrected_images[i], page_texts[i])
                    for i in retry_indices
                ]
                with ThreadPoolExecutor(max_workers=OCR_WORKERS) as pool:
                    retry_results = list(pool.map(_ocr_retry_page, retry_args))
                for idx, (full_text, corrected_img) in zip(retry_indices, retry_results):
                    page_texts[idx] = full_text
                    corrected_images[idx] = corrected_img
                # Re-run reference matching with the updated texts
                per_page_matches = find_references(
                    page_texts, reference_pattern, search_area
                )

        if not any(m is not None for m in per_page_matches):
            hint = ""
            if extraction_method == "ocr":
                hint = " (OCR was used)"
            elif extraction_method == "text" and not any(len(t.strip()) > 10 for t in page_texts):
                # Pages were empty — likely a scanned PDF
                if not OCR_AVAILABLE:
                    hint = (
                        ". This appears to be a scanned PDF. OCR is required but "
                        "poppler-utils is not installed. Install it with: "
                        "apt-get install poppler-utils (Linux) or "
                        "brew install poppler (macOS), or deploy with Docker "
                        "(see Dockerfile)"
                    )
                else:
                    hint = ". This may be a scanned PDF, try ocr='force'"
            return make_error(
                f"No references found matching pattern '{reference_pattern}'{hint}",
                "NO_REFERENCES_FOUND",
                404,
                pages_scanned=total_pages,
                extraction_method=extraction_method,
                ocr_available=OCR_AVAILABLE,
            )

        # 3. Group pages by reference
        page_groups, unmatched_pages = group_pages_by_reference(
            per_page_matches, total_pages, split_mode
        )

        # 3b. Include unmatched pages as "unknown"
        if unmatched_pages:
            page_groups["unknown"] = unmatched_pages

        # 4. Split the PDF
        split_pdfs = split_pdf(pdf_bytes, page_groups)

        # 5. Upload to GCS
        uploaded = upload_to_gcs(split_pdfs)

        # 6. Enrich with page info
        ref_to_pages = page_groups
        for doc in uploaded:
            pages = ref_to_pages.get(doc["reference"], [])
            doc["pages"] = pages
            doc["page_count"] = len(pages)

        return make_success(uploaded)

    except Exception as e:
        return make_error(f"Processing failed: {e}", "PROCESSING_ERROR", 500)
