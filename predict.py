import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("deepfake_model.h5")

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]
    if prediction > 0.5:
        return "FAKE", prediction
    else:
        return "REAL", 1 - prediction

result, confidence = predict_image("test.jpg")
print(result, confidence)
