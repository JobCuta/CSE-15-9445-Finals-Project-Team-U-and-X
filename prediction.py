import boto3
import numpy as np

import streamlit as st
import json

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

TAGS = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down',
               'clear', 'cloudy', 'conventional_mine', 'cultivation', 'habitation', 'haze', 'partly_cloudy',
               'primary', 'road', 'selective_logging', 'slash_burn', 'water']

DEFORESTATION_TAGS = ['agriculture', 'artisinal_mine', 'conventional_mine',
                      'cultivation', 'road', 'selective_logging', 'slash_burn']

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
    st.header('Prediction')
    pred_type = st.selectbox('Type of Prediction', ['User Upload', 'Random Image'])
    if pred_type == 'User Upload':
        
        with st.container():
            user_input = st.file_uploader(label='Upload an image', type='png',accept_multiple_files=False)
            # Gets prediction after the user uploads an image (Currently doesnt work with uploading images from the planet dataset)
            if user_input is not None:
                user_input = Image.open(user_input)
                user_input = user_input.resize((256, 256))
                user_input.save
                st.image(user_input)
                with st.spinner():
                    predictions = get_predictions('temp.png')
                # st.success(predictions)
    else:
        st.button('Predict from Test Data', on_click=random_prediction())



def random_prediction():
    # Test a supposed image
    # sample = ["test_9.jpg"]
    random_image = random.choice(prediction_lst)
    image = '{}/{}/{}'.format(URL, prediction_path, random_image)
    # Temporarily store the image in the folder so it can be referenced
    urllib.request.urlretrieve(image, 'temp.png')
    if image is not None:
        with st.container():
            col1, col2, col3 = st.columns([1,2,1])
            col2.image(image, use_column_width=True)
            with st.spinner():
                predictions = get_predictions('temp.png')
            # st.success(predictions)


# Runs the sagemaker runtime client to access the endpoint for inference
def get_predictions(image):
    endpoint = 'sagemaker-endpoint-v2'
    with Image.open(image) as image:
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format=image.format)
        img_byte_arr = img_byte_arr.getvalue()

        response = runtime.invoke_endpoint(EndpointName=endpoint,
                                           Body=img_byte_arr, ContentType='application/x-image')
        
        # Transforms the "response" to a readable array
        result_array = json.loads(response['Body'].read().decode())

        # Show "tags" above threshold (0.6)
        labels = []
        for tag, i in zip(TAGS, result_array):
            if i > 0.6:
                labels.append(tag)
                # print(tag, i)
        
        # Show labels in Streamlit
        show_amazon_labels(labels)

        # return response['Body'].read()

# Output label rows
def show_amazon_labels(labels, n_labels_per_row=4):
    n_labels = len(labels)
    is_there_def = 0
    if n_labels < n_labels_per_row:
        n_rows = 1
        n_final_cols = n_labels
        row = 1
    else:
        n_rows = n_labels // n_labels_per_row
        n_final_cols = n_labels % n_labels_per_row
        row = 0
    idx = 0
    while row <= n_rows:
        n_cols = n_labels_per_row
        if row == n_rows:
            if n_final_cols == 0:
                break
            else:
                n_cols = n_final_cols
        cols = st.columns(n_cols)
        for i in range(n_cols):
            with cols[i]:
                if labels[idx] in DEFORESTATION_TAGS:
                    st.error(labels[idx])
                    is_there_def = 1
                else:
                    st.success(labels[idx])
            idx += 1
        row += 1
        if is_there_def == 1:
            st.caption("The image shows forms of deforestation.")
        else: 
            st.caption("No forms of deforestation detected.")