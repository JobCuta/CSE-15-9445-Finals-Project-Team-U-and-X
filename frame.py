from pandas.core import frame
import app
import prediction
import streamlit as st
import metrics

# Pages displayed in the dropdown for page navigation
pages = {
    "Dashboard": app,
    "Prediction": prediction,
    "Metrics": metrics,
}

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select your page", tuple(pages.keys()))
pages[page].app()