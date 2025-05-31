

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],           # Allow all HTTP methods
    allow_headers=["*"],           # Allow all headers
)

# MODEL = tf.keras.models.load_model("../saved_models/1")
MODEL = tf.keras.layers.TFSMLayer("./saved_models/1", call_endpoint='serving_default')


CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    image = image.astype(np.float32) / 255.0
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL(img_batch)
    print("Predictions keys:", predictions.keys())  # Inspect keys
    print("Predictions:", predictions)

    # Extract actual prediction tensor by key (replace 'predictions' with your key)
    preds = predictions['dense_9']  
    preds = preds.numpy()

    predicted_class = CLASS_NAMES[np.argmax(preds[0])]
    confidence = np.max(preds[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

