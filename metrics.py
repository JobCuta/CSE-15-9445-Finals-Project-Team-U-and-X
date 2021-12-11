import streamlit as st
import streamlit.components.v1 as components

st.markdown(
    f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 1600px;
        padding-top: 2rem;
        padding-right: 5rem;
        padding-left: 5rem;
        padding-bottom: 2rem;
    }}
</style>
""",
    unsafe_allow_html=True,
)

def app():
    with st.container():
        st.header('Metrics of the Model')
        st.write('To properly display the line chart, change the Period to "Do not override" in the Options tab. Close the sidebar to properly get a view of metrics.')
        components.iframe("https://cloudwatch.amazonaws.com/dashboard.html?dashboard=model-metrics&context=eyJSIjoidXMtZWFzdC0xIiwiRCI6ImN3LWRiLTc2NzgwNjM4MTU2MSIsIlUiOiJ1cy1lYXN0LTFfc0xsUEtBbG5LIiwiQyI6ImtjOWFncHUwOHZpM2ljMjdnODJzaGxobCIsIkkiOiJ1cy1lYXN0LTE6YTY4NWE2NjAtZjViMC00M2IyLTlkOTQtYTY1MWM2YTNiODdmIiwiTyI6ImFybjphd3M6aWFtOjo3Njc4MDYzODE1NjE6cm9sZS9zZXJ2aWNlLXJvbGUvQ1dEQlNoYXJpbmctUHVibGljUmVhZE9ubHlBY2Nlc3MtUVpUSlg0WDAiLCJNIjoiUHVibGljIn0=",
        width= 1400, height=1300)
