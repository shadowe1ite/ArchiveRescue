# ArchiveRescue

A local-first tool for reading historical documents. It improves scans, runs OCR on your machine, highlights uncertain words and packages the results without overwriting the original.

## Features
- Enhancement (adaptive threshold, contrast or grayscale) and optional deskew
- Printed-text transcription using open-source Tesseract OCR
- Optional handwritten-line transcription with the open-weight TrOCR model
- Estimated OCR confidence and highlights for words that need review
- Editable text and downloadable ZIP with original, enhanced image, transcript and metadata

## Getting started

Python 3.10+ and Tesseract are required.

```bash
sudo apt install tesseract-ocr
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7860.

For experimental handwriting support, install `pip install -r requirements-handwriting.txt` before launching. The first model load downloads `microsoft/trocr-base-handwritten`; subsequent runs can use cached weights. TrOCR works best on cropped English handwriting lines.

## Why local?

Family records and archive scans can contain private information. ArchiveRescue uses open-source OCR and image processing on the user's machine rather than uploading documents to paid OCR APIs. No account or API key needed. To operate fully offline, install dependencies and download model weights beforehand.

## Important limitations

Image enhancement is a **derived** copy and may remove faint marks. The OCR text is not a reliable historical transcription without review. Confidence scores are estimates rather than probabilities. Handwriting recognition is experimental and does not provide confidence scores. Check dates, names, and unclear passages against the original. The ZIP notes that text may have been edited but is not an audit trail. Only expose the web UI on trusted networks.

## Who is it for?

Family historians, local museums, genealogists and freelance digitization services. A possible side hustle is charging for human-reviewed digitization, not unverified AI output.

## Tests

```bash
python -m unittest discover -s tests -v
```

MIT licensed. TrOCR and the upstream model weights retain their own licenses.
