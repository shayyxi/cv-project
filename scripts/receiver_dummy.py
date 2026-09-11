from fastapi import FastAPI, File, Form, UploadFile
import json

app = FastAPI()


@app.post("/api/detection-result")
async def receive_detection(
    metadata: str = Form(...),
    image: UploadFile = File(...),
):
    metadata_json = json.loads(metadata)
    image_bytes = await image.read()

    print("\n=== RECEIVED REQUEST ===")
    print("Metadata:")
    print(json.dumps(metadata_json, indent=2))

    print("\nImage:")
    print("filename:", image.filename)
    print("content_type:", image.content_type)
    print("size_bytes:", len(image_bytes))
    print("========================\n")

    return {
        "ok": True,
        "event_id": metadata_json.get("event_id"),
        "image_filename": image.filename,
        "image_size": len(image_bytes),
    }