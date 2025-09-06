import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('') #trained model

labels = ['cardboard', 'food and vegetation', 'glass','metal','paper', 'plastic', 'textiles','misc.']
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img = img.astype('float32') /255.0
    img = np.expand_dims(img, axis=0) #the size of the batch is 1 image

    #predict
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    label = labels[class_index]

    cv2.putText(frame, label, (200,200),cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0),2)
    cv2.imshow('Trash Model', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()















