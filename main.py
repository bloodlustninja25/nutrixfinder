import streamlit as st
import tensorflow as tf
import numpy as np

from keras.layers import InputLayer



def model_prediction(test_image):
    model = tf.keras.models.load_model('updated_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) 
    predictions = model.predict(input_arr)
    return np.argmax(predictions) 


st.sidebar.title("Sidebar")
app_mode = st.sidebar.selectbox("Select Page",["Home","About NutriFinder","Prediction"])


if(app_mode=="Home"):
    st.header("NutriFinder")


elif(app_mode=="About NutriFinder"):
    st.header("About NutriFinder")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items:")
    st.code("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")


elif(app_mode=="Prediction"):
    st.header("Model Prediction")
    test_image_cam = st.camera_input("Take a picture")
    if test_image_cam is not None:
        if(st.button("Show Image", key = "cam")):
            st.image(test_image_cam,width=4,use_column_width=True)
    test_image = test_image_cam
    test_image_up = st.file_uploader("Choose an Image:")
    if test_image_up is not None:
        test_image = test_image_cam
        if(st.button("Show Image", key = "upload")):
            st.image(test_image,width=4,use_column_width=True)
    
    if(st.button("Predict")):
        st.write("NutriFinder's Prediction")
        result_index = model_prediction(test_image)
        
        with open("labels.txt") as f:
            content = f.readlines()
        label = []
        for i in content:
            label.append(i[:-1])
        st.success("NutriFinder is Predicting it's a {}".format(label[result_index]))
