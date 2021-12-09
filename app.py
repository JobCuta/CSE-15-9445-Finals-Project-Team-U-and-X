import boto3
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

import random
import urllib.request
from PIL import Image
from io import BytesIO

from urllib.request import urlopen

# THINGS TO DO:
#  - ADD CHARTS FROM THE DATA EXPLORATION (done)
#  - SEPARATE THE DATA EXPLORATION AND PREDICTION PARTS (done)
#  - ADD PREPROCESSING TO THE USER UPLOADED IMAGE TO BE RESIZED TO (3, 256, 256)
#  - FIX THE APPEARANCE OF THE TRAINING AND VALIDATION DATA SAMPLE IMAGES (done)
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
prediction = 'image-classification/train-50-jpg'

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
    st.set_page_config(layout="wide")
    st.header('Dataset')
    st.dataframe(df)
    
    with st.container():
        st.header('Training Data')
        st.dataframe(train_df)

        choice = random.choice(train_sample)
 
        col1, col2, col3, col4, col5 = st.columns([1, 3, 0.2, 4, 1])

    with st.container():
        first, second = st.columns(2)
        
        with first:
            col2.write('**tags: {}**'.format(df.loc[choice.split('/')[-1]]['split_tags']))
            col2.image(choice, width = 500,
                    caption=choice.split('/')[-1])
        
        with second:
            col4.text(" ")
            col4.text(" ")
            
            prev = 0
            for i in range(3, 12, 4):
                col4.image(train_sample[prev:i + 1], width=160)
                prev = i + 1

    with st.container():
        st.header('Validation Data')
        st.dataframe(validation_df)
        
        choice = random.choice(validation_sample)
        col1, col2, col3, col4, col5 = st.columns([1, 3, 0.2, 4, 1])

        with st.container():
        
            first, second = st.columns(2)

            with first:
                col2.write('**tags: {}**'.format(df.loc[choice.split('/')[-1]]['split_tags']))
                col2.image(choice, width=500,caption=choice.split('/')[-1])

            with second:
                col4.text(" ")
                col4.text(" ")
                
                prev = 0
                for i in range(3, 12, 4):
                    col4.image(validation_sample[prev:i+1], width=160)
                    prev = i + 1

    with st.container():
        st.header('Data Exploration')

        #Plotting of pie chart
        fig = px.pie(df, names = 'present_tags', title = "Number of entries with deforestation tags")
        st.plotly_chart(fig, use_container_width=True)

        tag_df = pd.DataFrame(columns=['count','tag'])
        
        tag_columns = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 
        'clear', 'cloudy', 'conventional_mine', 'cultivation', 'habitation', 'haze', 'partly_cloudy', 
        'primary', 'road', 'selective_logging', 'slash_burn', 'water']
        
        tags_list = []
        
        for tag in tag_columns:
            tag_count = {}
            tag_count['tag'] = tag
            tag_count['count'] = df[tag].sum()
            tags_list.append(tag_count)
            
            final = tag_df.append(tags_list, ignore_index = True).sort_values(by='count', ascending=False)


        #Plotting of bar graph
        fig2 = px.bar(final, x='tag', y='count', title="Number of entries for each tag")
        st.plotly_chart(fig2, use_container_width=True)

    with st.container():

        column1, column2 = st.columns(2)

        with column1:

            training = px.pie(df[df['type'] == 'train'], names = 'present_tags', 
            title = "Number of entries with deforestation tags (Training Data)")
            column1.write(training)

        with column2:
            validation = px.pie(df[df['type'] == 'validation'], 
            names = 'present_tags', title = "Number of entries with deforestation tags (Validation Data)")
            column2.write(validation)

    with st.container():

        drowpdown = st.selectbox("Co-occurrence Matrix", 
        ["Matrix with labels","Matrix with weather labels"])

        def make_cooccurrence_matrix(labels):
            numeric_df = df[labels]
            c_matrix = numeric_df.T.dot(numeric_df)
            return c_matrix
        
        matrix1 = make_cooccurrence_matrix(tag_columns)

        if drowpdown == 'Matrix with labels':
            fig, ax = plt.subplots(figsize = (15,8))
            sns.heatmap(df.corr(), ax = ax)

            st.write(matrix1)
            st.write(fig)
        
        else:
            weather_labels = ['clear','partly_cloudy','haze','cloudy']
            matrix2 = make_cooccurrence_matrix(weather_labels)
            fig2, ax = plt.subplots(figsize = (15,8))
            sns.heatmap(matrix2.corr(), ax = ax)

            st.write(matrix2)
            st.write(fig2)
            
        
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
    image = '{}/{}/test_9.jpg'.format(URL, prediction)
    # Temporarily store the image in the folder so it can be referenced
    urllib.request.urlretrieve(image, 'temp.png')
    if image is not None:
        st.image(image, width=300)
        get_predictions('temp.png')
    
        
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
        payload = response['Body']
        preds = np.array(payload.read())
        print(preds)
        
if __name__ == '__main__':
    main()
