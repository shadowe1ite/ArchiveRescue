"""ArchiveRescue local web interface."""
import tempfile
from pathlib import Path

import gradio as gr
from PIL import Image

from archiverescue.core import Result, annotated, archive_package, recognize, restore


def process(image, method, deskew, engine, language):
    if image is None:
        raise gr.Error("Upload a scan first.")
    original = Image.fromarray(image).convert("RGB")
    enhanced = restore(original, method, deskew)
    try:
        result = recognize(enhanced, engine, language.strip() or "eng")
    except Exception as exc:
        raise gr.Error(f"OCR failed: {exc}") from exc

    confidence = "Unavailable" if result.confidence is None else f"{result.confidence:.1f}% (engine estimate)"
    status = f"**Engine:** {result.engine} | **Confidence:** {confidence}\n\n{result.warning}"
    info = dict(engine=result.engine, confidence=result.confidence,
                warning=result.warning, words=result.words)
    return enhanced, annotated(enhanced, result.words), result.text, status, info


def export(image, enhanced, transcript, metadata, method):
    if image is None or enhanced is None:
        raise gr.Error("Process a document before exporting.")
    metadata = metadata or {}
    result = Result(transcript or "", metadata.get("confidence"), metadata.get("engine", "unknown"),
                    metadata.get("warning", ""), metadata.get("words", []))
    bundle = archive_package(Image.fromarray(image).convert("RGB"),
                             Image.fromarray(enhanced).convert("RGB"),
                             transcript or "", result, "uploaded_document", method)
    destination = Path(tempfile.mkdtemp(prefix="archiverescue_")) / "archive_bundle.zip"
    destination.write_bytes(bundle)
    return str(destination)


with gr.Blocks(title="ArchiveRescue") as app:
    gr.Markdown("""# ArchiveRescue
### Preserve the original. Reveal the text.
A local-first workspace for cleaning up scanned records and checking AI transcriptions.
Documents stay on this computer rather than going to a paid OCR API.""")

    with gr.Row():
        with gr.Column():
            original = gr.Image(label="Original scan", type="numpy", sources=["upload", "clipboard"])
            method = gr.Radio(["Adaptive threshold", "High contrast", "Grayscale"],
                              value="Adaptive threshold", label="Image cleanup")
            deskew = gr.Checkbox(value=True, label="Straighten tilted scans")
            engine = gr.Radio(["Printed text (Tesseract)", "Handwriting (TrOCR, optional download)"],
                              value="Printed text (Tesseract)", label="OCR engine")
            language = gr.Textbox(value="eng", label="Tesseract language")
            run = gr.Button("Restore and transcribe", variant="primary")
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("Cleaned scan"):
                    enhanced = gr.Image(label="Enhanced copy", type="numpy", interactive=False)
                with gr.Tab("Words to check"):
                    marked = gr.Image(label="Low confidence words outlined", type="numpy", interactive=False)
            status = gr.Markdown("Choose a scan to start.")

    gr.Markdown("### Review the transcription")
    transcript = gr.Textbox(label="Edit before exporting", lines=12)
    metadata = gr.State({})
    save = gr.Button("Download archive bundle (.zip)")
    output = gr.File(label="Original scan, cleaned scan, corrected text and metadata")
    gr.Markdown("**Note:** OCR is not a substitute for checking historical source material. Handwriting mode works best with cropped English lines.")

    run.click(process, inputs=[original, method, deskew, engine, language],
              outputs=[enhanced, marked, transcript, status, metadata])
    save.click(export, inputs=[original, enhanced, transcript, metadata, method], outputs=output)


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7860)
