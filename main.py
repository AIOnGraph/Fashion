import streamlit as st
import joblib
from PIL import Image
import cv2
import numpy as np

model = joblib.load("cnn.joblib")

class_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

st.title("Fashion MNIST dataset Classification")

uploaded_image = st.file_uploader('Upload your image', type=['png', 'jpg', 'jpeg'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    grayscale_image = image.convert('L')
    img_array = np.array(grayscale_image)
    resized_img = cv2.resize(img_array, (28, 28))
    input_img = resized_img.reshape(1, 28, 28, 1)
    input_img = input_img / 255.0
    input_img = 1 - input_img  # Invert image (if necessary)
    
    prediction = model.predict(input_img)
    predicted_class = np.argmax(prediction)
    max_value = np.max(prediction)

    # Display prediction result with class label
    if predicted_class in class_labels:
        predicted_label = class_labels[predicted_class]
        st.write(f"Predicted Class:({predicted_label})")
        st.write('Prediction probability :{:.2f}%'.format(np.max(prediction)*100))
        st.write(f"Maximum Value: {max_value}")
    else:
        st.write("Unknown class label.")
