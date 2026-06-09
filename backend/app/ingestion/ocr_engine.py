"""
ingestion/ocr_engine.py
========================
Tesseract OCR engine for image-based PDF pages.

These BEE manuals are fully image-scanned (0 chars from native PDF text
extraction). This is the ONLY way to extract text from them.

Design:
- Renders each PDF page to PIL Image at 200 DPI (optimal for printed text)
- Applies preprocessing: grayscale + mild sharpening
- Runs Tesseract with --psm 1 (automatic page segmentation with OSD)
- Returns text + per-word confidence scores
- Falls back to --psm 3 if OSD fails
- Thread-safe (stateless)

Requirements:
    sudo apt-get install tesseract-ocr tesseract-ocr-eng
    pip install pytesseract pillow opencv-python-headless PyMuPDF
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image, ImageFilter

import logging

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
    return logger

logger = get_logger(__name__)

# Render DPI — 200 is optimal balance of OCR accuracy vs speed for printed text
OCR_DPI = 200
SCALE = OCR_DPI / 72.0  # PDF points → pixels


@dataclass
class OCRResult:
    """Result of OCR on a single PDF page."""
    page_num: int         # 0-indexed
    text: str             # Extracted text
    confidence: float     # Average Tesseract word confidence (0-100)
    word_count: int       # Number of words recognized
    ocr_time_ms: float    # Time taken
    has_content: bool     # Whether meaningful text was found


def _render_page(pdf_page: fitz.Page, dpi: int = OCR_DPI) -> Image.Image:
    """Render a PDF page to a PIL Image at the specified DPI."""
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    # Render as RGB
    pix = pdf_page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    img_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(img_bytes))


def _preprocess(img: Image.Image) -> Image.Image:
    """
    Preprocess image for better OCR accuracy.
    
    1. Convert to grayscale (reduces noise from color scanning)
    2. Apply mild sharpening (improves text edge definition)
    
    Note: We intentionally avoid aggressive binarization which can
    destroy low-contrast text in these scanned manuals.
    """
    gray = img.convert("L")
    sharpened = gray.filter(ImageFilter.SHARPEN)
    return sharpened


def _tesseract_ocr(img: Image.Image, psm: int = 1) -> tuple[str, float]:
    """
    Run Tesseract OCR on a PIL image.
    
    Args:
        img: Preprocessed grayscale or RGB image
        psm: Page segmentation mode (1=auto+OSD, 3=auto, 6=single block)
    
    Returns:
        Tuple of (text, avg_confidence)
    """
    try:
        import pytesseract
        
        config = f"--psm {psm} --oem 3 -l eng"
        
        # Get both text and confidence data
        data = pytesseract.image_to_data(
            img,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
        
        # Filter out empty/low-confidence words
        words = []
        confs = []
        for i, word in enumerate(data["text"]):
            w = word.strip()
            if w and data["conf"][i] != -1:
                words.append(w)
                confs.append(float(data["conf"][i]))
        
        text = pytesseract.image_to_string(img, config=config)
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        
        return text.strip(), avg_conf
        
    except Exception as e:
        logger.warning(f"Tesseract OCR failed with psm={psm}: {e}")
        return "", 0.0


def ocr_page(
    pdf_page: fitz.Page,
    page_num: int,
    dpi: int = OCR_DPI,
    min_confidence: float = 30.0,
) -> OCRResult:
    """
    Run OCR on a single PDF page.
    
    Strategy:
    1. Render at DPI
    2. Preprocess
    3. Try psm=1 (auto with OSD)
    4. If low confidence, retry with psm=3 (auto without OSD)
    5. Return best result
    
    Args:
        pdf_page: PyMuPDF Page object
        page_num: 0-indexed page number
        dpi: Render DPI
        min_confidence: Minimum confidence to accept psm=1 result
    
    Returns:
        OCRResult
    """
    t0 = time.monotonic()
    
    # Render
    img = _render_page(pdf_page, dpi)
    processed = _preprocess(img)
    
    # First pass: psm=1 (auto + OSD)
    text, confidence = _tesseract_ocr(processed, psm=1)
    
    # Retry with psm=3 if low confidence (OSD sometimes fails on book pages)
    if confidence < min_confidence and len(text) < 50:
        text2, confidence2 = _tesseract_ocr(processed, psm=3)
        if confidence2 > confidence or len(text2) > len(text):
            text, confidence = text2, confidence2
    
    elapsed_ms = (time.monotonic() - t0) * 1000
    word_count = len(text.split()) if text else 0
    has_content = word_count > 10 and confidence > 20.0
    
    return OCRResult(
        page_num=page_num,
        text=text,
        confidence=round(confidence, 1),
        word_count=word_count,
        ocr_time_ms=round(elapsed_ms, 1),
        has_content=has_content,
    )


def ocr_pdf_pages(
    pdf_path: str,
    page_nums: Optional[list[int]] = None,
    dpi: int = OCR_DPI,
    progress_callback=None,
) -> list[OCRResult]:
    """
    OCR multiple pages of a PDF.
    
    Args:
        pdf_path: Path to PDF file
        page_nums: List of 0-indexed page numbers (None = all pages)
        dpi: Render DPI
        progress_callback: Optional callable(current, total) for progress reporting
    
    Returns:
        List of OCRResult objects in page order
    """
    doc = fitz.open(str(pdf_path))
    total_pages = doc.page_count
    
    pages_to_ocr = page_nums if page_nums is not None else list(range(total_pages))
    results = []
    
    for i, pn in enumerate(pages_to_ocr):
        if pn >= total_pages:
            continue
        result = ocr_page(doc[pn], pn, dpi=dpi)
        results.append(result)
        
        if progress_callback:
            progress_callback(i + 1, len(pages_to_ocr))
    
    doc.close()
    return results
