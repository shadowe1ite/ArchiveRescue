"""Image cleanup, text recognition and archive packaging."""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageOps
from pytesseract import Output


@dataclass
class Result:
    text: str
    confidence: float | None
    engine: str
    warning: str
    words: list[dict]


def restore(image: Image.Image, method: str = "Adaptive threshold", deskew: bool = True) -> Image.Image:
    """Return a cleaned copy; never mutate the original."""
    rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=9)

    if deskew:
        ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(ink > 0))
        if len(coords) > 100:
            angle = cv2.minAreaRect(coords)[-1]
            angle = -(90 + angle) if angle < -45 else -angle
            if 0.2 < abs(angle) < 12:
                height, width = gray.shape
                rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
                gray = cv2.warpAffine(gray, rotation, (width, height), flags=cv2.INTER_CUBIC, borderValue=255)

    if method == "Adaptive threshold":
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12)
    elif method == "High contrast":
        gray = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8)).apply(gray)

    return Image.fromarray(gray)


def _printed_text(image: Image.Image, language: str) -> Result:
    data = pytesseract.image_to_data(image, lang=language, config="--psm 6", output_type=Output.DICT)
    words, confidence_values, lines = [], [], {}
    for i, token in enumerate(data["text"]):
        token = token.strip()
        if not token:
            continue
        confidence = float(data["conf"][i])
        line = f'{data["block_num"][i]}-{data["par_num"][i]}-{data["line_num"][i]}'
        word = {
            "text": token,
            "confidence": round(confidence, 1),
            "box": [int(data[key][i]) for key in ("left", "top", "width", "height")],
            "line": line,
        }
        words.append(word)
        lines.setdefault(line, []).append(token)
        if confidence >= 0:
            confidence_values.append(confidence)

    average = round(sum(confidence_values) / len(confidence_values), 1) if confidence_values else None
    warning = "OCR estimates can be wrong. Verify names, dates and unclear text against the original."
    return Result("\n".join(" ".join(line) for line in lines.values()), average, f"Tesseract ({language})", warning, words)


def _handwriting_lines(image: Image.Image) -> list[Image.Image]:
    gray = np.array(image.convert("L"))
    if gray.shape[1] > 1800:
        ratio = 1800 / gray.shape[1]
        gray = cv2.resize(gray, None, fx=ratio, fy=ratio)

    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, gray.shape[1] // 45), 3))
    merged = cv2.dilate(binary, kernel)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width > gray.shape[1] * 0.08 and 9 <= height < gray.shape[0] * 0.18:
            rects.append((x, y, width, height))

    rects.sort(key=lambda r: (r[1], r[0]))
    crops = []
    for x, y, width, height in rects[:80]:
        pad = max(5, height // 3)
        crop = gray[max(0, y - pad):min(gray.shape[0], y + height + pad),
                    max(0, x - pad):min(gray.shape[1], x + width + pad)]
        crops.append(Image.fromarray(crop).convert("RGB"))
    return crops


def recognize(image: Image.Image, engine: str = "Printed text (Tesseract)", language: str = "eng") -> Result:
    if engine.startswith("Printed"):
        return _printed_text(image, language)

    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import torch
    except ImportError as exc:
        raise RuntimeError("Install requirements-handwriting.txt to use handwriting mode") from exc

    model_name = "microsoft/trocr-base-handwritten"
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).eval()
    crops = _handwriting_lines(image)
    if not crops:
        return Result("", None, "TrOCR", "No lines detected. Try cropping a single handwriting line.", [])

    text = []
    for crop in crops:
        pixels = processor(images=crop, return_tensors="pt").pixel_values
        with torch.inference_mode():
            tokens = model.generate(pixels, max_new_tokens=128)
        text.append(processor.batch_decode(tokens, skip_special_tokens=True)[0])
    return Result("\n".join(text), None, "TrOCR handwriting",
                  "Handwriting recognition is experimental. Verify every reading against the original.", [])


def annotated(image: Image.Image, words: list[dict], threshold: float = 65) -> Image.Image:
    marked = image.convert("RGB").copy()
    drawing = ImageDraw.Draw(marked)
    for word in words:
        if word["confidence"] < threshold:
            x, y, width, height = word["box"]
            drawing.rectangle((x, y, x + width, y + height), outline="#d33d35", width=2)
    return marked


def archive_package(original: Image.Image, enhanced: Image.Image, text: str,
                    result: Result, source_name: str, method: str) -> bytes:
    metadata = {
        "source_filename": Path(source_name).name,
        "engine": result.engine,
        "confidence_estimate": result.confidence,
        "warning": result.warning,
        "enhancement": method,
        "transcription_edited_by_user": True,
        "word_bounding_boxes": result.words,
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, image in (("original.png", original), ("enhanced.png", enhanced)):
            buffer = io.BytesIO()
            image.save(buffer, "PNG")
            archive.writestr(filename, buffer.getvalue())
        archive.writestr("transcription.txt", text)
        archive.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
        archive.writestr("README.txt", "Keep the original scan. OCR requires manual verification.\n")
    return output.getvalue()
