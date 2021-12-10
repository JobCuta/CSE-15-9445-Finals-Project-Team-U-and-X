import boto3
import numpy as np

import streamlit as st

from PIL import Image
from io import BytesIO
import urllib.request
import random
from urllib.request import urlopen

# THINGS TO DO:
#  - ADD PREPROCESSING TO THE USER UPLOADED IMAGE TO BE RESIZED TO (3, 256, 256)
#  - ADD A FEATURE TO GET A RANDOM TEST IMAGE FROM THE S3 BUCKET TO RETURN A PREDICTION
#  - FIX ORDER OF PREDICTION (MAKE THE IMAGE AND PREDICTION APPEAR BELOW THE PREDICTION CONTAINER)
#  - FIGURE OUT THE THRESHOLD FOR THE TAG PREDICTIONS
#  - MAP THE PREDICTION TO THE TAG AND RETURN THE TAGS THAT ARE HIGHER THAN THE THRESHOLD
#       -> Optional: CHECK IF A DEFORESTATION TAG IS INCLUDED IN THE PREDICTION AND ADD AN INDICATION


URL = 'https://sagemaker-us-east-1-767806381561.s3.amazonaws.com'
s3_bucket = 'sagemaker-us-east-1-767806381561'
prediction_path = 'image-classification/train-50-jpg'
prediction_lst_path = 'image-classification/train-50-jpg-lst'

prediction_lst = []

# Client for runtime.sagemaker
runtime = boto3.client('runtime.sagemaker')

# Accessing the .lst file containing the files included in the train-50-jpg folder for user prediction
response = urlopen('{}/{}/test.lst'.format(URL, prediction_lst_path)).read()
for line in response.splitlines():
    prediction_lst.append(line.decode('utf-8').split('\t')[-1])

def app():
    st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 1400px;
        padding-top: 2rem;
        padding-right: 5rem;
        padding-left: 5rem;
        padding-bottom: 2rem;
    }}
</style>
""",
        unsafe_allow_html=True,
    )
    
    with st.container():
        st.header('Prediction')
        user_input = st.file_uploader(label='Upload an image', accept_multiple_files=False)

        # Gets prediction after the user uploads an image (Currently doesnt work with uploading images from the planet dataset)
        if user_input is not None:
            st.image(user_input, width=300)
            get_predictions(user_input)
            
        # Allow the user to predict from a random image (IMAGE NOT DISPLAYING AFTER PRESSING THE BUTTON, PREDICTIONS OKAY)
        st.button('Predict from Test Data', on_click=random_prediction)


def random_prediction():
    random_image = random.choice(prediction_lst)
    image = '{}/{}/{}'.format(URL, prediction_path, random_image)
    # Temporarily store the image in the folder so it can be referenced
    urllib.request.urlretrieve(image, 'temp.png')
    if image is not None:
        st.image(image, width=300)
        with st.spinner():
            predictions = get_predictions('temp.png')
        st.success(predictions)


# Runs the sagemaker runtime client to access the endpoint for inference
def get_predictions(image):
    endpoint = 'sagemaker-endpoint-v2'
    with Image.open(image) as image:
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format=image.format)
        img_byte_arr = img_byte_arr.getvalue()

        response = runtime.invoke_endpoint(EndpointName=endpoint,
                                                  Body=img_byte_arr, ContentType='application/x-image')

        # do not simplify into response['Body'].read(); will get error
        # do not simplify into response['Body'].read(); will get error
        payload = response['Body']
        return np.array(payload.read())
        
