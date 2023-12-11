import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from PIL import Image
import numpy as np

# Load the model architecture from JSON file
with open('xception_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)

# Load weights into the loaded model
loaded_model.load_weights("xception_weights.h5")

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize image to match Xception input size
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize pixel values between 0 and 1
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to perform classification
def classify_image(image):
    processed_img = preprocess_image(image)
    prediction = loaded_model.predict(processed_img)
    return prediction

# Streamlit app
st.title('Tomato Leaf Disease Classifier')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    if st.button('Classify'):
        with st.spinner('Classifying...'):
            prediction = classify_image(image)
            st.success('Classification done!')

        # Displaying the predicted class or classes (depending on your output)
        st.subheader('Prediction:')
        st.write(prediction)  # Modify this based on your model output format
