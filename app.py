import boto3
import streamlit as st
import pandas as pd
import numpy as np

import random
from PIL import Image
from io import BytesIO

from urllib.request import urlopen

# THINGS TO DO:
#  - ADD CHARTS FROM THE DATA EXPLORATION
#  - SEPARATE THE DATA EXPLORATION AND PREDICTION PARTS
#  - ADD PREPROCESSING TO THE USER UPLOADED IMAGE TO BE RESIZED TO (3, 256, 256)
#  - FIX THE APPEARANCE OF THE TRAINING AND VALIDATION DATA SAMPLE IMAGES 
#  - CLEAN THE CODE
#  - MAP THE PREDICTION TO THE TAG AND RETURN THE TAGS THAT ARE HIGHER THAN THE THRESHOLD
#       -> Optional: CHECK IF A DEFORESTATION TAG IS INCLUDED IN THE PREDICTION AND ADD AN INDICATION
#  - ADD A FEATURE TO GET A RANDOM TEST IMAGE FROM THE S3 BUCKET TO RETURN A PREDICTION

# URL and paths for S3
URL = 'https://sagemaker-us-east-1-767806381561.s3.amazonaws.com'
s3_bucket = 'sagemaker-us-east-1-767806381561'
train_path = 'image-classification/train'
train_lst_path = 'image-classification/train-lst'
validation_path = 'image-classification/validation'
validation_lst_path = 'image-classification/validation-lst'
random_prediction = 'image-classification/random-prediction'

# Client for runtime.sagemaker 
runtime = boto3.client('runtime.sagemaker')

# Preprocessed dataframe
df = pd.read_csv('{}/tagged_classes.csv'.format(URL), index_col=0)
train_df = df[df['type'] == 'train']
validation_df = df[df['type'] == 'validation']

# List for the files included in the training and validation dataset
train_lst = []    
validation_lst = []
tag_columns = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'clear', 'cloudy', 'conventional_mine',
               'cultivation', 'habitation', 'haze', 'partly_cloudy', 'primary', 'road', 'selective_logging', 'slash_burn', 'water']

# Accessing the .lst file containing the files included in the training data
response = urlopen('{}/{}/planet_train.lst'.format(URL, train_lst_path)).read()
for line in response.splitlines():
    train_lst.append(line.decode('utf-8').split('\t')[-2])


# Accessing the .lst file containing the files included in the validation data
response = urlopen('{}/{}/planet_validation.lst'.format(URL, validation_lst_path)).read()
for line in response.splitlines():
    validation_lst.append(line.decode('utf-8').split('\t')[-2])

# 25 random train images
train_sample = random.sample(train_lst, 15)
train_sample = ['{}/{}/{}'.format(URL, train_path, img) for img in train_sample]

# 25 random validation images
validation_sample = random.sample(validation_lst, 15) 
validation_sample = [
    '{}/{}/{}'.format(URL, validation_path, img) for img in validation_sample]

# example of how to get images from the S3 bucket
# st.image('{}/{}/train_1.jpg'.format(URL, train_path), width=200)

def main():
    st.header('Dataset')
    st.dataframe(df)
    
    with st.container():
        st.header('Training Data')
        st.dataframe(train_df)

        choice = random.choice(train_sample)
        col1, col2, col3 = st.columns([0.3, 0.7, 0.3])
        col2.write('tags: {}'.format(df.loc[choice.split('/')[-1]]['split_tags']))
        col2.image(choice, use_column_width=True,
                   caption=choice.split('/')[-1])
        
        prev = 0
        for i in range(4, 15, 5):
            st.image(train_sample[prev:i+1], width=140)
            prev = i + 1
            
    with st.container():
        st.header('Validation Data')
        st.dataframe(validation_df)
        
        choice = random.choice(validation_sample)
        col1, col2, col3 = st.columns([0.3, 0.7, 0.3])
        col2.write('tags: {}'.format(
            df.loc[choice.split('/')[-1]]['split_tags']))
        col2.image(choice, use_column_width=True,
                   caption=choice.split('/')[-1])
        
        prev = 0
        for i in range(4, 15, 5):
            st.image(validation_sample[prev:i+1], width=140)
            prev = i + 1
            
    with st.container():
        st.header('Prediction')
        user_input = st.file_uploader(label='Upload an image', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)

        # Gets prediction after the user uploads an image
        if user_input is not None:
            st.image(user_input, width=500)
            get_predictions(user_input)
            
# Runs the sagemaker runtime client to access the endpoint for inference
def get_predictions(image: Image):
    endpoint = 'sagemaker-endpoint-v2'
    with Image.open(image) as image:
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format=image.format)
        img_byte_arr = img_byte_arr.getvalue()

        response = runtime.invoke_endpoint(EndpointName=endpoint,
                                                  Body=img_byte_arr, ContentType='image/png')
        
        # do not simplify into response['Body'].read(); will get error
        payload = response['Body']
        preds = np.array(payload.read())
        print(preds)
        print(type(preds))
        
if __name__ == '__main__':
    main()
