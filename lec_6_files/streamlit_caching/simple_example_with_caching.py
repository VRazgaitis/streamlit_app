# https://docs.streamlit.io/library/advanced-features/caching#minimal-example
import streamlit as st
import pandas as pd

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)  # 👈 Download the data
    return df

df = load_data("https://github.com/plotly/datasets/raw/master/uber-rides-data1.csv")
st.dataframe(df)

st.button("Rerun")
