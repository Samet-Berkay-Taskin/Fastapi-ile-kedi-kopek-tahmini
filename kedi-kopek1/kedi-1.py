from keras.models import load_model
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
import uvicorn
import tensorflow as tf
from fastapi.responses import JSONResponse
import keras
from keras.preprocessing import image


new_model = load_model('kedi-kopek.h5')
new_model.summary()


app = FastAPI()



@app.get("/")
async def root():
    return {"Resimdeki kedi mi köpek mi ?"}


@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_arr = np.array(image, dtype=np.float32)
        resized = cv2.resize(image_arr, (64, 64), interpolation=cv2.INTER_AREA)
        test_foto1 = np.expand_dims(resized, axis=0)

        sonuc1 = new_model.predict(test_foto1)

        if sonuc1[0][0] > 0.5:
            Prediction = 'Kopek'
        else:
            Prediction = 'Kedi'

        return {Prediction}

    except Exception as e:
        print(str(e))
        return JSONResponse(content = {"status": "An error occurred."})
