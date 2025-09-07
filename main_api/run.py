from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load your trained model once at startup
MODEL_PATH = "model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Adjust this based on your training setup
IMG_SIZE = (224, 224)
CLASS_NAMES = ["cardboard", "food_organics", "glass", "metal",
               "misc_trash", "paper", "plastic", "textile", "vegetation"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize(IMG_SIZE)

    # 2. Preprocess
    img_array = np.array(image) / 255.0  # normalize if trained that way
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # 3. Prediction
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    predicted_class = CLASS_NAMES[predicted_class_idx]

    # 4. Return JSON result
    return {
        "predicted_class": predicted_class,
        "confidence": confidence
    }
