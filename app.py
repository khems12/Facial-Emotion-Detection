import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import os

model= tf.keras.models.load_model("Final_Resnet50_Best_model.keras")
 
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
    # Preprocess the image
    processed_image = prepare_image(image)
    # Make prediction using the model
    prediction = model.predict(processed_image)
    # Get the emotion label with the highest probability
    predicted_class = tf.argmax(prediction, axis=1).numpy()[0]
    predicted_emotion = emotion_labels.get(predicted_class, "Unknown Emotion")
    return predicted_emotion


# Set up Streamlit app layout
st.title("Facial Emotion Detection")
st.write("Upload a facial image and let the model predict the emotion.")

# Load the model

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
