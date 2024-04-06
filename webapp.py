
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from PIL import Image
import numpy as np

def load_model_json():
    with open('D:/DS course/Deploy/xception_model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    return loaded_model_json

loaded_model_json = load_model_json()

loaded_model = model_from_json(loaded_model_json)

# Load weights into the loaded model
loaded_model.load_weights("D:/DS course/Deploy/xception_weights.h5")

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
        # Define your class dictionary
        class_dict = {'Tomato_Bacterial_spot': 0, 'Tomato_Early_blight': 1, 'Tomato_Late_blight': 2, 'Tomato_Leaf_Mold': 3, 'Tomato_Septoria_leaf_spot': 4, 'Tomato_Spider_mites Two-spotted_spider_mite': 5, 'Tomato_Target_Spot': 6, 'Tomato_Tomato_Yellow_Leaf_Curl_Virus': 7, 'Tomato_Tomato_mosaic_virus': 8, 'Tomato_healthy': 9}

# Get the class name from the dictionary
        # Find the index of the maximum value in the prediction array
        prediction_index = np.argmax(prediction)

# Get the class name from the dictionary
        class_name = list(class_dict.keys())[list(class_dict.values()).index(prediction_index)]

        st.write(f"Predicted Class: {class_name}")
