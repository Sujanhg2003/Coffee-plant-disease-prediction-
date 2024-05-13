import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the saved CNN model
model = tf.keras.models.load_model(r"C:\Users\Sujan H G\Downloads\Coffe_Disease_002.h5")
class_labels = ['coffaenum', 'healthy', 'red spider mite', 'rust']

list=[]
# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to the size required by the model
    image = image.resize((224, 224))
    # Convert to numpy array
    img_array = np.array(image)
    # Normalize pixel values
    img_array = img_array / 255.0
    # Expand dimensions to match the input shape expected by the model
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    pred=np.argmax(prediction)
    label=class_labels[pred]
    confidence=np.max(prediction)
    list.append(label)
    list.append(confidence)




# Streamlit UI
st.title('COFFEE PLANT DISEASE DETECTION')
st.write('Upload an image to make predictions.')

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction when the 'Predict' button is clicked
    if st.button('Predict'):
        prediction = predict(image)
        st.write('Prediction:', list[0])
        st.write('Confidence:', list[1])
