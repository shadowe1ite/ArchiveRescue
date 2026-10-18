import io
import json
import unittest
import zipfile

from PIL import Image, ImageDraw, ImageFont
from archiverescue.core import archive_package, recognize, restore


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (1000, 220), "white")
        draw = ImageDraw.Draw(self.image)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 56)
        except OSError:
            font = ImageFont.load_default()
        draw.text((35, 65), "ARCHIVE RECORD 1892", fill="black", font=font)

    def test_original_is_unchanged(self):
        pixels = self.image.tobytes()
        self.assertEqual(restore(self.image).size, self.image.size)
        self.assertEqual(self.image.tobytes(), pixels)

    def test_ocr(self):
        result = recognize(self.image)
        self.assertIn("ARCHIVE", result.text)
        self.assertTrue(result.words)

    def test_archive_export(self):
        result = recognize(self.image)
        bundle = archive_package(self.image, restore(self.image), "edited", result, "sample.png", "Grayscale")
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            self.assertEqual(archive.read("transcription.txt").decode(), "edited")
            self.assertIn("original.png", archive.namelist())
            self.assertEqual(json.loads(archive.read("metadata.json"))["engine"], "Tesseract (eng)")


if __name__ == "__main__":
    unittest.main()
