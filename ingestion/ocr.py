"""OCR fallback for scanned/image-based PDF pages that have no extractable text layer.

Lazy-loads a single shared `easyocr.Reader` (loading its detection + recognition models is slow
and only needed at all when a source actually turns out to be a scan), rather than importing
easyocr or constructing a Reader at module import time.
"""

import numpy as np

_READER = None


def _get_reader():
    global _READER
    if _READER is None:
        import easyocr

        _READER = easyocr.Reader(["en", "fr"], gpu=False, verbose=False)
    return _READER


def ocr_page(fitz_page, dpi: int = 300) -> list[str]:
    """Render a PyMuPDF page to a raster image and OCR it, returning one string per detected
    text line (unmerged, so the caller's line-clustering/boilerplate logic can treat OCR output
    the same way as a real text layer's extracted lines)."""
    pix = fitz_page.get_pixmap(dpi=dpi)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return _get_reader().readtext(image, detail=0, paragraph=False)
