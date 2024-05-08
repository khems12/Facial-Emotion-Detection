import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import requests
import os

# Function to download the model file from Google Drive
# def download_model():
#     file_id = "1B1U1n0y4Dxxi18zw68N8byV0VNpXLa5y"  # Update with your file ID
#     url = f"https://drive.google.com/uc?id={file_id}"
#     response = requests.get(url)
#     with open("best_model.keras", "wb") as f:
#         f.write(response.content)

import requests

def download_model():
    # file_id = "1B1U1n0y4Dxxi18zw68N8byV0VNpXLa5y"  # Update with your file ID
    # url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get("https://drive.google.com/file/d/1B1U1n0y4Dxxi18zw68N8byV0VNpXLa5y/view?usp=drive_link")
    with open("Final_Resnet50_Best_model.keras", "wb") as f:
        f.write(response.content)


# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        return tf.keras.models.load_model("Final_Resnet50_Best_model.keras")
    except Exception as e:
        st.error("Error loading model. Please make sure the model file is correct.")
        st.error(e)
        return None

# Emotion labels dictionary
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def prepare_image(img):
    """Preprocess the image to fit your model's input requirements."""
    # Resize the image to the target size
    img = img.resize((224, 224))
    # Convert the image to a numpy array
    img_array = img_to_array(img)
    # Expand dimensions to match the model's input shape
    img_array = tf.expand_dims(img_array, axis=0)
    # Rescale pixel values to [0,1], as done during training
    img_array /= 255.0
    return img_array

def predict_emotion(image, model):
    print("Inside predict_emotion function")
    # Preprocess the image
    processed_image = prepare_image(image)
    # Make prediction using the model
    prediction = model.predict(processed_image)
    print("Prediction:", prediction)
    # Get the emotion label with the highest probability
    predicted_class = tf.argmax(prediction, axis=1).numpy()[0]
    predicted_emotion = emotion_labels.get(predicted_class, "Unknown Emotion")
    print("Predicted emotion:", predicted_emotion)
    return predicted_emotion


# Set up Streamlit app layout
st.title("Facial Emotion Detection")
st.write("Upload a facial image and let the model predict the emotion.")

# Check if model file exists, if not, download it
if not os.path.isfile("Final_Resnet50_Best_model.keras"):
    with st.spinner("Downloading model..."):
        download_model()

# Load the model
model = load_model()

# Define function to handle image upload and prediction
def classify_emotion():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # Make prediction if model is loaded successfully
        if model is not None:
            predicted_emotion = predict_emotion(image, model)
            # Display prediction
            st.write("### Predicted Emotion:")
            st.write(predicted_emotion)

# Display the emotion classification widget
classify_emotion()
