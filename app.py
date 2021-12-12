import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

import random
from urllib.request import urlopen

# THINGS TO DO:
#  - ADD CHARTS FROM THE DATA EXPLORATION (done)
#  - SEPARATE THE DATA EXPLORATION AND PREDICTION PARTS (done)
#  - FIX THE APPEARANCE OF THE TRAINING AND VALIDATION DATA SAMPLE IMAGES (done)
#  - FIX THE APPEARANCE OF THE DASHBOARD CHARTS (RESIZE CHARTS? REORGANIZE COMPONENTS?)
#  - FIX THE PROCESSING SPEED FOR CHANGING THE DASHBOARD CHARTS (CACHE?)

# URL and paths for S3
URL = 'https://sagemaker-us-east-1-767806381561.s3.amazonaws.com'
s3_bucket = 'sagemaker-us-east-1-767806381561'
train_path = 'image-classification/train'
train_lst_path = 'image-classification/train-lst'
validation_path = 'image-classification/validation'
validation_lst_path = 'image-classification/validation-lst'

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
# train_sample = random.sample(train_lst, 15)
# train_sample = ['{}/{}/{}'.format(URL, train_path, img) for img in train_sample]

# 25 random validation images
validation_sample = random.sample(validation_lst, 15) 
validation_sample = [
    '{}/{}/{}'.format(URL, validation_path, img) for img in validation_sample]

# example of how to get images from the S3 bucket
# st.image('{}/{}/train_1.jpg'.format(URL, train_path), width=200)

def app():
    # st.set_page_config(layout="wide")
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
    st.header('Dataset')
    st.dataframe(df)
    
    with st.container():
        st.header('Training Data')
        st.dataframe(train_df)
        st.subheader('Sample images of Training Data')
        st.write('A maximum of 50 sample images will be displayed.')
        tags = st.multiselect('Select Tags', train_df.columns[2:-2]) #selected tags
        #convert tags to list 
        #search train_df for sublist of selected tags 
        
        # tags_df = train_df[train_df['tags'].isin(tags)]
        tags_df = train_df['tags']
        tags_list = []
        for row in train_df.itertuples():
            result = all(tag in row.tags for tag in tags)
            if result==True:
                tags_list.append(row)
        tags_df = pd.DataFrame(tags_list)
        tags_df = tags_df.set_index(tags_df['Index'])
        # st.write(tags_df.index, 'tagslist')
        

        tags_df = ['{}/{}/{}'.format(URL, train_path, img) for img in tags_df.index] #convert image names to links
        if len(tags) != 0:
            if len(tags_df) != len(train_df['tags']):
                st.write(f'A total number of **{len(tags_df)}** images were found with those tags.')
                with st.container():
                    i_col1, i_col2, i_col3, i_col4, i_col5, i_col6, i_col7, i_col8, i_col9, i_col10 = st.columns(10)
                    i_cols = [i_col1, i_col2, i_col3, i_col4, i_col5, i_col6, i_col7, i_col8, i_col9, i_col10]
                    prev = 0
                    curr_icol = 0
                    for i in tags_df:
                        i_cols[curr_icol].image(i, width=100)
                        prev = prev + 1
                        if prev == 50: #maximum images to show
                            break
                        if curr_icol == 9:
                            curr_icol = 0
                        else:
                            curr_icol = curr_icol + 1
            else:
                st.write('No image with those tags cannot be found.')

    with st.container():
        st.header('Validation Data')
        st.dataframe(validation_df)

        with st.container():
            choice = random.choice(validation_sample)
            col1, col2, col3 = st.columns([3, 0.2, 4])

            col1.write(
                '**tags: {}**'.format(df.loc[choice.split('/')[-1]]['split_tags']))
            col1.image(choice, width=500,
                       caption=choice.split('/')[-1])

            col3.text(" ")
            col3.text(" ")
            prev = 0
            for i in range(3, 12, 4):
                col3.image(validation_sample[prev:i + 1], width=165)
                prev = i + 1


    with st.container():
        st.header('Data Exploration')
        col1, col2 = st.columns([0.5, 0.5])
        
        with col1:
            deforestation_dropdown = st.selectbox('Deforestation Sample Split', ['Dataset', 'Training Data', 'Validation Data'])
            if deforestation_dropdown == 'Dataset':
                #Plotting of pie chart
                fig = px.pie(df, names = 'present_tags', title = "Number of entries with deforestation tags")
                st.plotly_chart(fig, use_container_width=True)
            elif deforestation_dropdown == 'Training Data':
                training = px.pie(df[df['type'] == 'train'], names='present_tags',
                                  title="Number of entries with deforestation tags (Training Data)")
                st.write(training)
            else:
                validation = px.pie(df[df['type'] == 'validation'],
                                    names='present_tags', title="Number of entries with deforestation tags (Validation Data)")
                st.write(validation)
                
            
        with col2:
            tag_dropdown = st.selectbox('Tags', ['Dataset', 'Training Data', 'Validation Data'])
            if tag_dropdown == 'Dataset':
                fig = px.bar(get_tag_counts(df), x='tag', y='count', title="Number of entries for each tag")
                st.plotly_chart(fig, use_container_width=True)
            elif tag_dropdown == 'Training Data':
                training = px.bar(get_tag_counts(train_df), x='tag', y='count',
                              title="Number of entries for each tag (Training Data)")
                st.plotly_chart(training, use_container_width=True)
            else:
                validation = px.bar(get_tag_counts(validation_df), x='tag', y='count',
                              title="Number of entries for each tag (Validation Data)")
                st.plotly_chart(validation, use_container_width=True)
    
    
    with st.container():
        drowpdown = st.selectbox("Co-occurrence Matrix", 
        ["Matrix with all labels","Matrix with weather labels", "Matrix with other tags"])

        def make_cooccurrence_matrix(labels):
            numeric_df = df[labels]
            c_matrix = numeric_df.T.dot(numeric_df)
            return c_matrix
        
        matrix1 = make_cooccurrence_matrix(tag_columns)

        if drowpdown == 'Matrix with all labels':
            fig, ax = plt.subplots(figsize = (15,8))
            sns.heatmap(df.corr(), ax = ax)

            st.write(matrix1)
            st.write(fig)
        
        elif drowpdown == 'Matrix with weather labels':
            weather_labels = ['clear','partly_cloudy','haze','cloudy']
            matrix2 = make_cooccurrence_matrix(weather_labels)
            fig2, ax = plt.subplots(figsize = (15,8))
            sns.heatmap(matrix2.corr(), ax = ax)

            st.write(matrix2)
            st.write(fig2)
        else:
            other_labels = ['agriculture','artisinal_mine','bare_ground','blooming','blow_down','conventional_mine','cultivation','habitation','primary','road','selective_logging','slash_burn','water']
            matrix3 = make_cooccurrence_matrix(other_labels)
            fig3, ax = plt.subplots(figsize = (15,8))
            sns.heatmap(matrix3.corr(), ax = ax)
            st.write(matrix3)
            st.write(fig3)

# Function to get the sum of the tags for a given dataset
def get_tag_counts(dataframe):
    tag_df = pd.DataFrame(columns=['tag', 'count'])
    tag_columns = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down',
                   'clear', 'cloudy', 'conventional_mine', 'cultivation', 'habitation', 'haze', 'partly_cloudy',
                   'primary', 'road', 'selective_logging', 'slash_burn', 'water']

    tags_list = []
    for tag in tag_columns:
        tag_count = {}
        tag_count['tag'] = tag
        tag_count['count'] = dataframe[tag].sum()
        tags_list.append(tag_count)
    tag_df = tag_df.append(tags_list, ignore_index=True).sort_values(by='count', ascending=False)
    return tag_df
