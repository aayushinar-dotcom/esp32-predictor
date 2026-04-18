from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Same model you're already using — downloads automatically, no file needed
model = YOLO("yolov8n.pt")

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image sent by ESP32 (replaces your cv2.VideoCapture)
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Same inference you're already doing — just on ESP32 image instead of webcam
        results = model(image, imgsz=320)

        detections = []
        for result in results:
            for box in result.boxes:
                label = model.names[int(box.cls)]
                confidence = round(float(box.conf), 2)
                detections.append({
                    "label": label,
                    "confidence": confidence
                })

        if detections:
            # Best detection (highest confidence)
            best = max(detections, key=lambda x: x["confidence"])
            return JSONResponse({
                "status": "ok",
                "prediction": best["label"],
                "confidence": best["confidence"],
                "all_detections": detections  # all objects found in the image
            })
        else:
            return JSONResponse({
                "status": "ok",
                "prediction": "nothing detected",
                "confidence": 0.0,
                "all_detections": []
            })

    except Exception as e:
        return JSONResponse({"error": str(e), "status": "error"}, status_code=500)