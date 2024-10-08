

import pandas as pd

import pyarrow.parquet as pq
import streamlit as st


df = pq.read_table(source="data/mushrooms.parquet").to_pandas()
st.dataframe(st.session_state.df.head(5))