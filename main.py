from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import io
from PIL import Image

app = FastAPI(title="AletheiaModel API", description="Clasificador de alimentos")

# Cargar modelo y embeddings
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
example_embeddings = np.load("./models/example_embeddings.npy")
example_labels = np.load("./models/example_labels.npy")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Leer imagen subida
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((224, 224))

        # Preprocesar y generar embedding
        x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
        vec = model.predict(x)[0]

        # Calcular similitudes
        sims = cosine_similarity([vec], example_embeddings)[0]
        idx = np.argmax(sims)

        return JSONResponse({
            "label": str(example_labels[idx]),
            "confidence": float(sims[idx])
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"message": "AletheiaModel API is running!"}
