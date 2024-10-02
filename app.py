import streamlit as st
from PIL import Image
import pickle
import numpy as np
import io
import joblib
import cv2
from utils import image_processings as ip

# Load the trained classifier model from the pickle file
with open('artifacts/svm_model.pkl', 'rb') as f:
    model = joblib.load(f)

# Entrepreneur name mapping
entrepreneur_mapping = {
    0: "Bill Gates",
    1: "Elon Musk",
    2: "Irfan Malik",
    3: "Jack Ma",
    4: "Ritesh Agarwal",
    5: "Steve Jobs"
}

# Function for making a prediction (replace with your class predict logic)
def predict_entrepreneur(image):
    # Preprocess the image
    faces_image = ip.get_visible_faces(image, is_path=False)[0]
    processed_image = ip.pre_process_image(faces_image, 64, is_path=False)
    
    # Assuming the model expects a numpy array of the processed image
    prediction = model.predict([processed_image])
    
    return prediction

# Streamlit UI
st.title("Entrepreneur Image Classifier")
st.write("Upload an image of an entrepreneur, and the model will predict their name!")

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], )

# Display image and make prediction
if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Add a spinner while processing
    with st.spinner("Processing the image..."):
        # Convert the image to bytes and use it for preprocessing and prediction
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        np_img = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        # Make a prediction using your model
        prediction = predict_entrepreneur(image)
    
    # Map the prediction to the entrepreneur name
    predicted_class = prediction[0]
    predicted_name = entrepreneur_mapping.get(predicted_class, "Unknown")
    
    # Display the predicted name
    st.success(f"Entrepreneur name is: {predicted_name}")

# Footer or additional information
st.markdown("""
---
Built with ❤️ by Ali Bin Kashif
""")
