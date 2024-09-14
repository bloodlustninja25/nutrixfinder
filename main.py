import streamlit as st
import tensorflow as tf
import numpy as np
from keras.layers import InputLayer
import requests
import pandas as pd
import google.generativeai as genai
import os
import sys
import config

API_URL = "https://trackapi.nutritionix.com/v2/natural/nutrients"
APP_ID = "d6dcf9c3"  # Replace with your Nutritionix app_id
APP_KEY = "1bdeefeb98c792ce9de0c4f3c15197cd"  # Replace with your Nutritionix app_key
GOOGLE_API_KEY = "AIzaSyDxgSJn7VwIMQCYMFG1dNGYulARUgEYVrY"

genai.configure(api_key="AIzaSyDxgSJn7VwIMQCYMFG1dNGYulARUgEYVrY")

model = genai.GenerativeModel("gemini-1.5-flash")

def get_nutritional_info(item):
    headers = {
        'x-app-id': APP_ID,
        'x-app-key': APP_KEY,
        'Content-Type': 'application/json'
    }
    data = {
        "query": item,
        "timezone": "US/Eastern"
    }
    
    response = requests.post(API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: Unable to retrieve nutritional information for {item}")
        return None


def model_prediction(test_image):
    model = tf.keras.models.load_model('updated_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    
    # Get prediction probabilities
    predictions = model.predict(input_arr)
    
    # Get the predicted class and the confidence score
    predicted_class = np.argmax(predictions)
    confidence_score = np.max(predictions)  # Get the highest probability (confidence score)
    
    return predicted_class, confidence_score

def get_diet_recommendation(predicted_item, health_goal, GOOGLE_API_KEY):
    response = model.generate_content(f"Create a balanced diet plan based on the following criteria: - **Health Goal**: {health_goal} - **Predicted Food**: {predicted_item}. Please provide a detailed meal plan that aligns with the specified health goal and also include veg options. Also, indicate whether the predicted food should be included in the diet plan or excluded, and provide reasons for this recommendation. Include alternative food suggestions if the predicted food is excluded. For the diet plan: - Suggest meals and snacks. - Include portion sizes and frequency of consumption. - Ensure the plan is nutritionally balanced and meets the health goal specified. Stick to Indian recipes and measurment units")
    return response.text
    

st.sidebar.title("Sidebar")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About NutriFinder", "Prediction"])

if app_mode == "Home":
    st.header("NutriFinder")

elif app_mode == "About NutriFinder":
    st.header("About NutriFinder")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items:")
    st.code("fruits - banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("vegetables - cucumber, carrot, capsicum, onion, potato, lemon, tomato, radish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalapeño, ginger, garlic, peas, eggplant.")
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")

elif app_mode == "Prediction":
    st.header("Model Prediction")

    # Initialize session state variables if not present
if "test_image" not in st.session_state:
    st.session_state.test_image = None

if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False  # Track if the prediction has been made

if "predicted_class" not in st.session_state:
    st.session_state.predicted_class = None

if "confidence_score" not in st.session_state:
    st.session_state.confidence_score = None

if "health_goal" not in st.session_state:
    st.session_state.health_goal = "Select your health goal"

# Ask the user if they want to use the camera or upload an image
use_camera = st.radio("How would you like to provide the image?", ("Upload an Image", "Use Camera"))

if use_camera == "Use Camera":
    st.info("Please turn on your camera to capture an image.")
    test_image_cam = st.camera_input("Take a picture")
    if test_image_cam is not None:
        st.session_state.test_image = test_image_cam  # Save the image in session state
        if st.button("Show Image", key="cam"):
            st.image(st.session_state.test_image, use_column_width=True)

elif use_camera == "Upload an Image":
    test_image_up = st.file_uploader("Choose an Image:")
    if test_image_up is not None:
        st.session_state.test_image = test_image_up  # Save the image in session state
        if st.button("Show Image", key="upload"):
            st.image(st.session_state.test_image, use_column_width=True)

# Only show the Predict button if an image is uploaded/captured
if st.session_state.test_image is not None:
    if st.button("Predict"):
        # Run the model prediction
        st.write("NutriFinder's Prediction")
        result_index, confidence_score = model_prediction(st.session_state.test_image)

        with open("labels.txt") as f:
                content = f.readlines()
        label = [i.strip() for i in content]  # Use .strip() to remove newline characters

        # Save the prediction and confidence score in session state
        st.session_state.predicted_item = label[result_index]
        st.session_state.confidence_score = confidence_score
        st.session_state.prediction_made = True  # Mark that prediction has been made

# If prediction is made, show the results and health goal selection
if st.session_state.prediction_made:
    # Display the prediction and confidence score
    predicted_item = st.session_state.predicted_item
    confidence_score = st.session_state.confidence_score

    st.success(f"NutriFinder is predicting it's a {predicted_item} with {confidence_score * 100:.2f}% confidence.")
    
    # Capture the user's health goal using st.radio() here, and store it in session state
    st.subheader("Personalized Diet Recommendation")
    st.session_state.health_goal = st.radio("Select Your Health Goal", 
                                            ("Select your health goal", "Muscle Building", "Fat Loss", "Weight Gain"), 
                                            index=("Select your health goal", "Muscle Building", "Fat Loss", "Weight Gain").index(st.session_state.health_goal))
    
    if st.session_state.health_goal == "Select your health goal":
        st.warning("Please select a valid health goal to proceed.")
    else:
        # Generate diet recommendation based on the predicted food item and health goal
        recommendation = get_diet_recommendation(st.session_state.predicted_item, st.session_state.health_goal, GOOGLE_API_KEY)
        st.write(f"Based on your goal for {st.session_state.health_goal}, here's a balanced diet recommendation:")
        st.write(recommendation)

else:
    st.error("Please upload or capture an image before making a prediction.")
