# Imports

import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import pandas as pd
import google.generativeai as genai
from gtts import gTTS
import tempfile
import re
from cv2 import cv2
import os
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import io
from PIL import Image

# Nutritionix and Genai Config

API_URL = "https://trackapi.nutritionix.com/v2/natural/nutrients"
APP_ID = "d6dcf9c3" 
APP_KEY = "1bdeefeb98c792ce9de0c4f3c15197cd"
GOOGLE_API_KEY = "AIzaSyDxgSJn7VwIMQCYMFG1dNGYulARUgEYVrY"


genai.configure(api_key="AIzaSyDxgSJn7VwIMQCYMFG1dNGYulARUgEYVrY")

model = genai.GenerativeModel("gemini-1.5-flash")

IMG_SIZE = 400
LR = 1e-3
no_of_fruits=7


# For TTS
def clean_text(text):
    # Remove markdown special characters like #, *, and excess whitespace
    clean_text = re.sub(r'[#*]', '', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # Remove excess spaces
    return clean_text

# access nutritionix api
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
    
# Get area of the food

#image_segment
def getAreaOfFood(image):
    data=os.path.join(os.getcwd(),"images")
    if os.path.exists(data):
        print('folder exist for images at ',data)
    else:
        os.mkdir(data)
        print('folder created for images at ',data)
        
    cv2.imwrite('{}\\1 original image.jpg'.format(data),image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('{}\\2 original image BGR2GRAY.jpg'.format(data),img)
    img_filt = cv2.medianBlur( img, 5)
    cv2.imwrite('{}\\3 img_filt.jpg'.format(data),img_filt)
    img_th = cv2.adaptiveThreshold(img_filt,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2)
    cv2.imwrite('{}\\4 img_th.jpg'.format(data),img_th)
    contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #make change here


	# find contours. sort. and find the biggest contour. the biggest contour corresponds to the plate and fruit.
    mask = np.zeros(img.shape, np.uint8)
    largest_areas = sorted(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest_areas[-1]], 0, (255,255,255,255), -1)
    cv2.imwrite('{}\\5 mask.jpg'.format(data),mask)
    img_bigcontour = cv2.bitwise_and(image,image,mask = mask)
    cv2.imwrite('{}\\6 img_bigcontour.jpg'.format(data),img_bigcontour)

	# convert to hsv. otsu threshold in s to remove plate
    hsv_img = cv2.cvtColor(img_bigcontour, cv2.COLOR_BGR2HSV)
    cv2.imwrite('{}\\7 hsv_img.jpg'.format(data),hsv_img)
    h,s,v = cv2.split(hsv_img)
    mask_plate = cv2.inRange(hsv_img, np.array([0,0,50]), np.array([200,90,250]))
    cv2.imwrite('{}\\8 mask_plate.jpg'.format(data),mask_plate)
    mask_not_plate = cv2.bitwise_not(mask_plate)
    cv2.imwrite('{}\\9 mask_not_plate.jpg'.format(data),mask_not_plate)
    fruit_skin = cv2.bitwise_and(img_bigcontour,img_bigcontour,mask = mask_not_plate)
    cv2.imwrite('{}\\10 fruit_skin.jpg'.format(data),fruit_skin)

	#convert to hsv to detect and remove skin pixels
    hsv_img = cv2.cvtColor(fruit_skin, cv2.COLOR_BGR2HSV)
    cv2.imwrite('{}\\11 hsv_img.jpg'.format(data),hsv_img)
    skin = cv2.inRange(hsv_img, np.array([0,10,60]), np.array([10,160,255])) #Scalar(0, 10, 60), Scalar(20, 150, 255)
    cv2.imwrite('{}\\12 skin.jpg'.format(data),skin)
    not_skin = cv2.bitwise_not(skin); #invert skin and black
    cv2.imwrite('{}\\13 not_skin.jpg'.format(data),not_skin)
    fruit = cv2.bitwise_and(fruit_skin,fruit_skin,mask = not_skin) #get only fruit pixels
    cv2.imwrite('{}\\14 fruit.jpg'.format(data),fruit)
    
    fruit_bw = cv2.cvtColor(fruit, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('{}\\15 fruit_bw.jpg'.format(data),fruit_bw)
    fruit_bin = cv2.inRange(fruit_bw, 10, 255) #binary of fruit
    cv2.imwrite('{}\\16 fruit_bw.jpg'.format(data),fruit_bin)

	#erode before finding contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    erode_fruit = cv2.erode(fruit_bin,kernel,iterations = 1)
    cv2.imwrite('{}\\17 erode_fruit.jpg'.format(data),erode_fruit)

	#find largest contour since that will be the fruit
    img_th = cv2.adaptiveThreshold(erode_fruit,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    cv2.imwrite('{}\\18 img_th.jpg'.format(data),img_th)
    contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask_fruit = np.zeros(fruit_bin.shape, np.uint8)
    largest_areas = sorted(contours, key=cv2.contourArea)
    cv2.drawContours(mask_fruit, [largest_areas[-2]], 0, (255,255,255), -1)
    cv2.imwrite('{}\\19 mask_fruit.jpg'.format(data),mask_fruit)

	#dilate now
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask_fruit2 = cv2.dilate(mask_fruit,kernel2,iterations = 1)
    cv2.imwrite('{}\\20 mask_fruit2.jpg'.format(data),mask_fruit2)
    fruit_final = cv2.bitwise_and(image,image,mask = mask_fruit2)
    cv2.imwrite('{}\\21 fruit_final.jpg'.format(data),fruit_final)
    
	#find area of fruit
    img_th = cv2.adaptiveThreshold(mask_fruit2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    cv2.imwrite('{}\\22 img_th.jpg'.format(data),img_th)
    contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    largest_areas = sorted(contours, key=cv2.contourArea)
    fruit_contour = largest_areas[-2]
    fruit_area = cv2.contourArea(fruit_contour)

	
	#finding the area of skin. find area of biggest contour
    skin2 = skin - mask_fruit2
    cv2.imwrite('{}\\23 skin2.jpg'.format(data),skin2)
	#erode before finding contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    skin_e = cv2.erode(skin2,kernel,iterations = 1)
    cv2.imwrite('{}\\24 skin_e .jpg'.format(data),skin_e )
    img_th = cv2.adaptiveThreshold(skin_e,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    cv2.imwrite('{}\\25 img_th.jpg'.format(data),img_th)
    contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask_skin = np.zeros(skin.shape, np.uint8)
    largest_areas = sorted(contours, key=cv2.contourArea)
    cv2.drawContours(mask_skin, [largest_areas[-2]], 0, (255,255,255), -1)
    cv2.imwrite('{}\\26 mask_skin.jpg'.format(data),mask_skin)
    
    
    skin_rect = cv2.minAreaRect(largest_areas[-2])
    box = cv2.boxPoints(skin_rect)
    box = np.int0(box)
    mask_skin2 = np.zeros(skin.shape, np.uint8)
    cv2.drawContours(mask_skin2,[box],0,(255,255,255), -1)
    cv2.imwrite('{}\\27 mask_skin2.jpg'.format(data),mask_skin2)
    
    pix_height = max(skin_rect[1])
    pix_to_cm_multiplier = 5.0/pix_height
    skin_area = cv2.contourArea(box)
    
    
    return fruit_area,fruit_bin ,fruit_final,skin_area, fruit_contour, pix_to_cm_multiplier

density_dict = { 1:0.609, 2:0.94, 3:0.641,  4:0.641,5:0.513, 6:0.482,7:0.481}

skin_multiplier = 5*2.3

def getVolume(label, area, skin_area, pix_to_cm_multiplier, fruit_contour):
	area_fruit = (area/skin_area)*skin_multiplier #area in cm^2
	label = int(label)
	volume = 100
	if label == 1 or label == 5 or label == 7 or label == 6 : #sphere-apple,tomato,orange,kiwi,onion
		radius = np.sqrt(area_fruit/np.pi)
		volume = (4/3)*np.pi*radius*radius*radius
		#print (area_fruit, radius, volume, skin_area)
	
	if label == 2 or label == 4 or (label == 3 and area_fruit > 30): #cylinder like banana, cucumber, carrot
		fruit_rect = cv2.minAreaRect(fruit_contour)
		height = max(fruit_rect[1])*pix_to_cm_multiplier
		radius = area_fruit/(2.0*height)
		volume = np.pi*radius*radius*height
		
	if (label==4 and area_fruit < 30) : # carrot
		volume = area_fruit*0.5 #assuming width = 0.5 cm
	
	return volume

def getMass(label, volume): #volume in cm^3
	density = density_dict[int(label)]
	mass = volume*density*1.0
	return mass #mass

def mass_main(result,img):
    img_path =img
    fruit_areas,final_f,areaod,skin_areas, fruit_contours, pix_cm = getAreaOfFood(img_path)
    volume = getVolume(result, fruit_areas, skin_areas, pix_cm, fruit_contours)
    mass = getMass(result, volume)
    fruit_volumes=volume
    fruit_mass=mass
    return fruit_mass

def get_model(IMG_SIZE,no_of_fruits,LR):
	try:
		tf.reset_default_graph()
	except:
		print("tensorflow")
	convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

	convnet = conv_2d(convnet, 32, 5, activation='relu')

	convnet = max_pool_2d(convnet, 5)

	convnet = conv_2d(convnet, 64, 5, activation='relu')

	convnet = max_pool_2d(convnet, 5)

	convnet = conv_2d(convnet, 128, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)

	convnet = conv_2d(convnet, 64, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)


	convnet = conv_2d(convnet, 32, 5, activation='relu')
	convnet = max_pool_2d(convnet, 5)

	convnet = fully_connected(convnet, 1024, activation='relu')
	convnet = dropout(convnet, 0.8)

	convnet = fully_connected(convnet, no_of_fruits, activation='softmax')
	convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

	model = tflearn.DNN(convnet, tensorboard_dir='log')

	return model

def get_name_mass(test_data):
    MODEL_NAME = 'Fruits_dectector-{}-{}.model'.format(LR, '5conv-basic')

    model_save_at=os.path.join("model",MODEL_NAME)

    model=get_model(IMG_SIZE,no_of_fruits,LR)

    model.load(model_save_at)
    labels=list(np.load('labels.npy'))
    
    img=cv2.imread(test_data)
    img1=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    model_out=model.predict([img1])
    result=np.argmax(model_out)
    name=labels[result]
    mass=round(mass(result+1,img),2)
    
    return name, mass



def get_diet_recommendation(predicted_item, health_goal, GOOGLE_API_KEY):
    response = model.generate_content(f"Create a balanced diet plan based on the following criteria: - **Health Goal**: {health_goal} - **Predicted Food**: {predicted_item}. Please provide a detailed meal plan that aligns with the specified health goal and also include veg options. Also, indicate whether the predicted food should be included in the diet plan or excluded, and provide reasons for this recommendation. Include alternative food suggestions if the predicted food is excluded. For the diet plan: - Suggest meals and snacks. - Include portion sizes and frequency of consumption. - Ensure the plan is nutritionally balanced and meets the health goal specified. Stick to Indian recipes and measurement units.")
    return response.text
    


# Streamlit part

if "test_image" not in st.session_state:
    st.session_state.test_image = None

if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False  

if "predicted_item" not in st.session_state:
    st.session_state.predicted_item = None

if "mass" not in st.session_state:
    st.session_state.mass = None

if "health_goal" not in st.session_state:
    st.session_state.health_goal = "Select your health goal"

# Sidebar selection for pages
st.sidebar.title("Sidebar")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About NutriFinder", "Prediction"])

if app_mode == "Home":
    st.header("NutriFinder")
    st.write("Welcome to NutriFinder! A place to get nutrition information and personalized diet recommendations.")

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

    if use_camera == "Use Camera":
        st.info("Please turn on your camera to capture an image.")
        test_image_cam = st.camera_input("Take a picture")
        if test_image_cam is not None:
            st.session_state.test_image = test_image_cam 
            if st.button("Show Image", key="cam"):
                st.image(st.session_state.test_image, use_column_width=True)

    elif use_camera == "Upload an Image":
        test_image_up = st.file_uploader("Choose an Image:")
        if test_image_up is not None:
            st.session_state.test_image = test_image_up  
            if st.button("Show Image", key="upload"):
                st.image(st.session_state.test_image, use_column_width=True)

    if st.session_state.test_image is not None:
        if isinstance(st.session_state.test_image, bytes):  # Camera input gives bytes
                img = Image.open(io.BytesIO(st.session_state.test_image))
        else:  # File uploader gives a file-like object
                img = Image.open(st.session_state.test_image)
        img_array = np.array(img)

        if st.button("Predict"):
            st.write("NutriFinder's Prediction")
            
            result_name, result_mass = get_name_mass(img_array)

            st.session_state.predicted_item = result_name
            st.session_state.mass = result_mass
            st.session_state.prediction_made = True

    if st.session_state.prediction_made:

        st.success(f"NutriFinder is predicting it's a {st.session_state.predicted_item} of {st.session_state.mass:.2f} gms mass.")
        nutrition_data = get_nutritional_info(st.session_state.predicted_item, result_mass)
        if nutrition_data:
                st.subheader(f"Nutritional Information for {st.session_state.predicted_item.capitalize()}:")
                for food in nutrition_data['foods']:
                    serving_qty = food['serving_qty']
                    serving_unit = food['serving_unit']
                    serving_weight = food['serving_weight_grams']

                    nutrition_info = {
                        "Nutrient": ["Serving Size", "Calories (kcal)", "Total Fat (g)", "Carbohydrates (g)", "Protein (g)"],
                        "Value": [
                            f"{serving_qty} {serving_unit} ({serving_weight} g)",
                            round(food['nf_calories'] / serving_weight * result_mass, 2),
                            round(food['nf_total_fat'] / serving_weight * result_mass, 2),
                            round(food['nf_total_carbohydrate'] / serving_weight * result_mass, 2),
                            round(food['nf_protein'] / serving_weight * result_mass, 2)
                        ]
                    }

                    nutrition_df = pd.DataFrame(nutrition_info)

                    
                    st.table(nutrition_df)
        st.subheader("Personalized Diet Recommendation")
        st.session_state.health_goal = st.radio("Select Your Health Goal", 
                                                ("Select your health goal", "Muscle Building", "Fat Loss", "Weight Gain"), 
                                                index=("Select your health goal", "Muscle Building", "Fat Loss", "Weight Gain").index(st.session_state.health_goal))
        
    if st.session_state.health_goal == "Select your health goal":
        st.warning("Please select a valid health goal to proceed.")
    else:
        recommendation = get_diet_recommendation(st.session_state.predicted_item, st.session_state.health_goal, GOOGLE_API_KEY)
        st.write(f"Based on your goal for {st.session_state.health_goal}, here's a balanced diet recommendation:")
        st.write(recommendation)

        output_text = f"NutriFinder is predicting it's a {st.session_state.predicted_item} with {st.session_state.confidence_score * 100:.2f}% confidence." + recommendation
        cleaned_output_text = clean_text(output_text)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            tts = gTTS(cleaned_output_text, slow=False)
            tts.save(temp_audio_file.name)
            audio_bytes = temp_audio_file.read()
            st.markdown("## Your diet recommendation in audio format:")
            st.audio(temp_audio_file.name, format="audio/mp3")