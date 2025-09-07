import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import numpy as np
import tensorflow as tf
from PIL import Image
import io

from starlette.staticfiles import StaticFiles

app = FastAPI()

model = tf.keras.models.load_model("model.h5")

classes = ['cardboard', 'food', 'glass','metal','paper', 'plastic', 'textiles','misc']

app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the HTML upload page
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" href="/static/style.css">
        <title>Trash Classifier</title>
    </head>
    <body>
        <h2 id="upload">Upload Trash Image</h2>
        <input type="file" id="fileInput" accept="image/*">
        <button onclick="uploadImage()">Predict</button>
        <p id="result"></p>

        <script>
        async function uploadImage() {
            const fileInput = document.getElementById("fileInput");
            if (fileInput.files.length === 0) {
                alert("Please select a file!");
                return;
            }

            const formData = new FormData();
            formData.append("file", fileInput.files[0]);

            const response = await fetch("/predict/", {
                method: "POST",
                body: formData
            });

            const result = await response.json();
            document.getElementById("result").innerText =
                `Prediction: ${result.class} (confidence: ${(result.confidence * 100).toFixed(2)}%)`;
        }
        </script>
    </body>
    </html>
    """

@app.get("/")
def read_root():
    return {"message": "Send a POST request to /predict with an image file."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read file bytes
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Preprocess: resize to match training input size (e.g. 224x224)
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0  # normalize like training
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension

    # Run prediction
    preds = model.predict(img_array)
    predicted_class = classes[np.argmax(preds[0])]
    confidence = float(np.max(preds[0]))

    return {"class": predicted_class, "confidence": confidence}


if __name__ == "__main__":
    uvicorn.run("fast_apiiii:app", host="0.0.0.0", port=8000, reload=True)

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}
#
# if __name__ == "__main__":
#     uvicorn.run("fast_apiiii:app", host="0.0.0.0", port=8000, reload=True)
