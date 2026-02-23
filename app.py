from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import io
from PIL import Image
import gdown
import os

app = FastAPI(title="Image Classifier API")

# Загрузка модели
MODEL_PATH = "model/best_model.keras"
os.makedirs("model", exist_ok=True)
if not os.path.exists(MODEL_PATH):
    gdown.download(
        "https://drive.google.com/uc?id=1089S8paExVJWgjMLROu16rOnGF1upA-9",
        MODEL_PATH,
        quiet=False,
        fuzzy=True
    )
model = load_model(MODEL_PATH)

# Классы
classes = {0: 'Capsicum', 1: 'Carrot', 2: 'Cucumber', 3: 'Potato', 4: 'Tomato'}

INPUT_SIZE = (224, 224)

def preprocess_image(file: UploadFile) -> np.ndarray:
    """Читает и преобразует изображение в формат, пригодный для модели"""
    image_bytes = file.file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(INPUT_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, 3)
    return img_array

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        img_array = preprocess_image(file)
        predictions = model.predict(img_array)
        probs = predictions[0]
        class_idx = int(np.argmax(probs))
        class_label = classes[class_idx]

        return JSONResponse({
            "predicted_class": class_label,
            "probabilities": {
                label: float(prob) for label, prob in zip(classes.values(), probs)
            }
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

