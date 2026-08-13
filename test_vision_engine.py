

from pathlib import Path

import cv2

from app.processing.cv.ppe_vision_engine import PPEVisionEngine
from app.processing.cv.Vision_renderer import VisionRenderer


# ============================================================
# Configuration
# ============================================================

IMAGE_PATH = Path(
    "A7Gallion_Cam1 - 20221202160651.JPG"
)

OUTPUT_PATH = Path(
    "vision_engine_test_result.jpg"
)


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Read input image
    # --------------------------------------------------------

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {IMAGE_PATH}"
        )

    print(
        f"Input image: "
        f"{image.shape[1]}x{image.shape[0]}"
    )


    # --------------------------------------------------------
    # 2. Convert image to bytes
    #
    # This is exactly what the backend will eventually provide
    # to process_image().
    # --------------------------------------------------------

    success, encoded = cv2.imencode(
        ".jpg",
        image,
    )

    if not success:
        raise RuntimeError(
            "Failed to encode input image."
        )

    image_bytes = encoded.tobytes()


    # --------------------------------------------------------
    # 3. Create Vision Engine
    #
    # Models are loaded internally by the engine.
    # --------------------------------------------------------

    engine = PPEVisionEngine()


    # --------------------------------------------------------
    # 4. Run vision pipeline
    # --------------------------------------------------------

    result = engine.process_image(
        image_bytes
    )


    # --------------------------------------------------------
    # 5. Print results
    # --------------------------------------------------------

    print(
        f"\nWorkers detected: "
        f"{result.worker_count}"
    )

    for person in result.detections:

        print(
            f"\nPerson {person.person_id}"
        )

        print(
            f"  Confidence: "
            f"{person.confidence:.3f}"
        )

        print(
            f"  Bounding box: "
            f"{person.box}"
        )

        print(
            f"  Helmet: "
            f"{person.compliance.helmet}"
        )

        print(
            f"  Vest: "
            f"{person.compliance.vest}"
        )

        print(
            f"  Boots: "
            f"{person.compliance.boots}"
        )

        print(
            f"  Compliant: "
            f"{person.compliance.compliant}"
        )

        print(
            f"  PPE detections: "
            f"{len(person.ppe)}"
        )

        for ppe in person.ppe:

            print(
                f"    - "
                f"{ppe.label}: "
                f"{ppe.confidence:.3f} "
                f"{ppe.box}"
            )


    # --------------------------------------------------------
    # 6. Render annotations
    #
    # IMPORTANT:
    # The renderer receives the result from the engine.
    # It does NOT run inference.
    # --------------------------------------------------------

    renderer = VisionRenderer()

    annotated_image = renderer.draw_original(
        image,
        result,
    )


    # --------------------------------------------------------
    # 7. Save annotated image
    # --------------------------------------------------------

    success = cv2.imwrite(
        str(OUTPUT_PATH),
        annotated_image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    )

    if not success:
        raise RuntimeError(
            f"Failed to save output image: "
            f"{OUTPUT_PATH}"
        )

    print(
        f"\nAnnotated image saved to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()

