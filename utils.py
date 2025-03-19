import os
import pandas as pd
from pathlib import Path
import pytz
import streamlit as st

# Constants
DATA_DIR = Path(__file__).parent / 'data'
CACHE_DIR = Path(__file__).parent / 'cached_data'

# Create cache directory if it doesn't exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Function to get cache filename
def get_cache_filename(csv_filename, timeframe_code):
    base_name = os.path.splitext(csv_filename)[0]
    return f"{base_name}_{timeframe_code.replace(':', '_')}.pkl"

# Function to convert timestamps to NY local time
def convert_to_ny_local_time(df, timestamp_column='datetime'):
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    df[timestamp_column] = df[timestamp_column] - pd.Timedelta(hours=1)
    utc_tz = pytz.UTC
    ny_tz = pytz.timezone('America/New_York')
    df[timestamp_column] = df[timestamp_column].dt.tz_localize(utc_tz)
    df[timestamp_column] = df[timestamp_column].dt.tz_convert(ny_tz)
    df[timestamp_column] = df[timestamp_column].dt.tz_localize(None)
    return df

def download_high_probability_results(high_prob_results, selected_tf_code):
    filtered_csv_data = high_prob_results.to_csv(index=False)
    st.download_button(
        label="Download High Probability Results as CSV",
        data=filtered_csv_data,
        file_name=f"high_probability_{selected_tf_code}_results.csv",
        mime="text/csv"
    )