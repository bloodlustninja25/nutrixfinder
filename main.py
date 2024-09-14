import streamlit as st
import tensorflow as tf
import numpy as np
from keras.layers import InputLayer
import requests
import pandas as pd
import google.generativeai as genai
import config.py

genai.configure(api_key=GOOGLE_API_KEY)
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

    # Ask the user if they want to use the camera or upload an image
    use_camera = st.radio("How would you like to provide the image?", ("Upload an Image", "Use Camera"))

    test_image = None

    if use_camera == "Use Camera":
        # Ask the user to turn on the camera
        st.info("Please turn on your camera to capture an image.")
        test_image_cam = st.camera_input("Take a picture")
        if test_image_cam is not None:
            test_image = test_image_cam
            if st.button("Show Image", key="cam"):
                st.image(test_image, use_column_width=True)

    elif use_camera == "Upload an Image":
        # Allow the user to upload an image
        test_image_up = st.file_uploader("Choose an Image:")
        if test_image_up is not None:
            test_image = test_image_up
            if st.button("Show Image", key="upload"):
                st.image(test_image, use_column_width=True)

    # Ensure test_image is not None before trying to predict
    if test_image is not None and st.button("Predict"):
        st.write("NutriFinder's Prediction")
        result_index, confidence_score = model_prediction(test_image)

        # Check if the confidence score is above or below 50%
        if confidence_score < 0.5:
            st.warning("The confidence score is below 50%. Please upload or capture a clearer image for better prediction.")
        else:
            with open("labels.txt") as f:
                content = f.readlines()
            label = [i.strip() for i in content]  # Use .strip() to remove newline characters

            predicted_item = label[result_index]
            st.success(f"NutriFinder is predicting it's a {predicted_item} with {confidence_score*100:.2f}% confidence.")
            
            # Get nutritional information for the predicted item
            nutrition_data = get_nutritional_info(predicted_item)
            if nutrition_data:
                st.subheader(f"Nutritional Information for {predicted_item.capitalize()}:")
                
                # Display key nutritional values
                for food in nutrition_data['foods']:
                    serving_qty = food['serving_qty']
                    serving_unit = food['serving_unit']
                    serving_weight = food['serving_weight_grams']
                    
                    # Create a dictionary with nutritional values
                    nutrition_info = {
                        "Nutrient": ["Serving Size", "Calories (kcal)", "Total Fat (g)", "Carbohydrates (g)", "Protein (g)"],
                        "Value": [
                            f"{serving_qty} {serving_unit} ({serving_weight} g)",
                            food['nf_calories'],
                            food['nf_total_fat'],
                            food['nf_total_carbohydrate'],
                            food['nf_protein']
                        ]
                    }
                    
                    # Convert the dictionary into a DataFrame for a better table display
                    nutrition_df = pd.DataFrame(nutrition_info)
                    
                    # Display the table
                    st.table(nutrition_df.style.hide(axis="index"))
            st.subheader("Personalized Diet Recommendation")
            health_goal = "Muscle Building"
            health_goal = st.radio("Select Your Health Goal", ("Muscle Building", "Fat Loss", "Weight Gain"))

            # Generate diet recommendation based on the predicted food item and health goal
            recommendation = get_diet_recommendation(predicted_item, health_goal, GOOGLE_API_KEY)
            st.write(f"Based on your goal for {health_goal}, here's a balanced diet recommendation:")
            st.write(recommendation)
    else:
        st.error("Please upload or capture an image before making a prediction.")