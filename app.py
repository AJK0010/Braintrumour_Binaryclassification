import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

# Install Git LFS (run once)
!git lfs install 

# Tell Git LFS to track the large file
!git lfs track "brain_tumor.h5" 

# Add and commit the files
!git add .
!git commit -m "Initial commit with model tracked by LFS"

# Push to GitHub
git push origin main

# --- Configuration ---
# FIX: Changed the model filename to the correct H5 format 
# that was used and supported by Keras (as seen in the PDF)
MODEL_FILENAME = 'brain_tumor.h5' 
# The image size used for training the model
IMG_SIZE = 224

# --- Function to Load and Preprocess Image ---
def preprocess_image(uploaded_file):
    """
    Loads an image from the uploaded file, resizes it, converts it to an array,
    normalizes it, and expands dimensions for model input.
    """
    # Load image using PIL
    img = Image.open(uploaded_file).convert('RGB')
    
    # Resize the image to the target size (224x224)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to create a batch of size 1 (required by Keras model)
    # Shape changes from (224, 224, 3) to (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the pixel values (as done during training: img_array / 255)
    img_array /= 255.0
    
    return img, img_array

# --- Streamlit App Layout ---
def main():
    st.title("🧠 Brain Tumor Detection using CNN")
    st.markdown("Upload an MRI scan to predict whether a brain tumor is present.")
    
    # --- Load Model ---
    try:
        # Load the model saved in the PDF
        model = load_model(MODEL_FILENAME)
        st.success(f"Model '{MODEL_FILENAME}' loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # The first error (No such file) will appear if 'brain_tumor.h5' is missing.
        st.warning("Please ensure the model file **'brain_tumor.h5'** is in the same directory as this script.")
        # Stop execution if model loading fails
        st.stop()
        
    # --- File Uploader ---
    uploaded_file = st.file_uploader(
        "Choose a brain MRI image...", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        # Show a spinner while processing
        with st.spinner('Processing image and predicting...'):
            # Preprocess the image
            display_img, processed_img = preprocess_image(uploaded_file)
            
            # Display the uploaded image
            st.image(display_img, caption='Uploaded MRI Scan', use_column_width=True)
            st.write("")
            
            # --- Make Prediction ---
            try:
                # Prediction step (similar to the PDF: model.predict(img_array))
                prediction = model.predict(processed_img)
                
                # Get the scalar prediction value
                prediction_value = prediction[0][0]
                
                # Apply the decision rule from the PDF (if prediction >= 0.5)
                if prediction_value >= 0.5:
                    result = "🚫 **Positive: You have a brain tumor.**"
                    st.error(result)
                else:
                    result = "✅ **Negative: You do not have a brain tumor.**"
                    st.success(result)
                
                # Display the raw prediction value for transparency
                st.info(f"Raw Model Confidence Score: **{prediction_value:.4f}**")
                
                st.markdown(
                    """
                    > **Note:** This is a prediction from a trained model and **is not a substitute for professional medical advice.**
                    """
                )
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

if __name__ == '__main__':
    main()
