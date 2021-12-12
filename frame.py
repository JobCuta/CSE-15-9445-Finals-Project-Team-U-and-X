from pandas.core import frame
import app
import prediction
import streamlit as st
import metrics
import index
# Pages displayed in the dropdown for page navigation
pages = {
    "About": index,
    "Dashboard": app,
    "Metrics": metrics,
    "Prediction": prediction,
}

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select your page", tuple(pages.keys()))
pages[page].app()
