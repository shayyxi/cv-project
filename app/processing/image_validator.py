from io import BytesIO

from PIL import Image

from app.exceptions import ValidationError


class ImageValidator:
    def validate(self, image_bytes: bytes) -> None:
        if not image_bytes:
            raise ValidationError("Image bytes are empty.")

        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image.verify()
        except Exception as exc:
            raise ValidationError("Invalid or corrupted image.") from exc