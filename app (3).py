import streamlit as st
from keras.applications.mobilenet_v2 import MobileNetV2
import numpy as np
from keras.applications.mobilenet_v2 import preprocess_input,decode_predictions
from PIL import Image
import cv2

model = MobileNetV2()

st.title("Image Classifier")
upload = st.sidebar.file_uploader(label='Upload the Image')
if upload is not None:
  file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
  opencv_image = cv2.imdecode(file_bytes, 1)
  opencv_image = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2RGB)
  img = Image.open(upload)
  st.image(img,caption='Uploaded Image',width=300)

  if st.sidebar.button('PREDICT'):
    st.sidebar.write("Result:")
    x = cv2.resize(opencv_image,(224,224))
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    y = model.predict(x)
    label = decode_predictions(y)
    # print the classification
    for i in range(3):
      out = label[0][i]
      st.sidebar.title('%s (%.2f%%)' % (out[1], out[2]*100))
    