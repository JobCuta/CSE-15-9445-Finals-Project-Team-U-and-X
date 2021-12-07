import streamlit as st
import pandas as pd
import random
from urllib.request import urlopen

s3_bucket = 'sagemaker-us-east-1-767806381561'
URL = 'https://sagemaker-us-east-1-767806381561.s3.amazonaws.com'

train_path = 'image-classification/train'
train_lst_path = 'image-classification/train-lst'
validation_path = 'image-classification/validation'
validation_lst_path = 'image-classification/validation-lst'

df = pd.read_csv('{}/tagged_classes.csv'.format(URL))
train_lst = []    
validation_lst = []

response = urlopen('{}/{}/planet_train.lst'.format(URL, train_lst_path)).read()
for line in response.splitlines():
    train_lst.append(line.decode('utf-8').split('\t')[-2])
print(len(train_lst))

response = urlopen('{}/{}/planet_validation.lst'.format(URL, validation_lst_path)).read()
for line in response.splitlines():
    validation_lst.append(line.decode('utf-8').split('\t')[-2])
print(len(validation_lst))

train_sample = random.sample(train_lst, 25)  # 25 random train images
validation_sample = random.sample(validation_lst, 25) # 25 random validation images

# st.image('{}/{}/train_1.jpg'.format(URL, train_path), width=100)