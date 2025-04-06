import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import io
import os
import pickle
import swifter  # For parallel processing
import pytz
import utils
import data_preparation
import analysis
import display


# Streamlit configuration
st.set_page_config(layout="wide", page_title="Financial Data Analysis")
st.title("Financial Price Movement Probability Analysis")
st.markdown("""
This app analyzes financial futures data to determine probability of price movements.
For each H1 candle, it checks if the first M5 candle closes above the H1 opening price,
then calculates the probability of a 0.1% move against the direction of the M5 close before
a XX% move in the opposite direction.
""")

# Sidebar parameters
tp_percent = st.sidebar.number_input("Take Profit %", value=0.1, step=0.05, format="%.2f")
sl_percent = st.sidebar.number_input("Stop Loss %", value=0.3, step=0.05, format="%.2f")


# Reference timeframe selection
reference_timeframe_options = ["1 Minute (M1)", "5 Minutes (M5)","10 Minutes (M10)", "15 Minutes (M15)", "30 Minutes (M30)"]
reference_timeframe_mapping = {
    "1 Minute (M1)": "1T",
    "5 Minutes (M5)": "5T",
    "10 Minutes (M10)": "10T",
    "15 Minutes (M15)": "15T",
    "30 Minutes (M30)": "30T"
}

selected_reference_tf = st.sidebar.selectbox(
    "Select reference timeframe",
    options=reference_timeframe_options,
    index=1  # Default to M5
)

selected_reference_tf_code = reference_timeframe_mapping[selected_reference_tf]

# Timeframe selection for resampling
# This is not needed so commenting it out and defaulting selected_tf to H1
#resampling_options = ["5 Minutes (M5)", "15 Minutes (M15)", "30 Minutes (M30)", "1 Hour (H1)", "4 Hours (H4)"]
resample_mapping = {
    "5 Minutes (M5)": "5T",
    "15 Minutes (M15)": "15T",
    "30 Minutes (M30)": "30T",
    "1 Hour (H1)": "1H",
    "4 Hours (H4)": "4H"
}

#selected_tf = st.sidebar.selectbox(
#    "Select larger timeframe",
#    options=resampling_options,
#    index=3  # Default to H1
#)
selected_tf = "1 Hour (H1)"
selected_tf_code = "1H"


# Toggle switch for end-of-timeframe restriction
enable_end_of_tf_restriction = st.sidebar.checkbox(
    "Classify scenarios as losers if neither target nor stop level is hit by the end of the larger timeframe",
    value=False  # Default to disabled
)

# Toggle switch for trade direction
enable_reverse_calculation = st.sidebar.checkbox(
    "Calculate the probabilities with fading the reference candle",
    value=False  # Default to disabled
)

# Multiselect widget for days of the week
selected_days = st.sidebar.multiselect(
    "Select days of the week to include in the analysis",
    options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Sunday'],
    default=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Sunday']  # Default to weekdays
)

prepare_data = st.sidebar.button("Prepare & Cache Data")
start_analysis = st.sidebar.button("Start Analysis")
filter_best_candles = st.sidebar.button("Filter by Best H1 Candles")

# Main function to run the analysis
def run_analysis(selected_tf_code, selected_reference_tf_code, tp_percent, sl_percent, enable_end_of_tf_restriction, enable_reverse_calculation, selected_days):
    selected_files = list(utils.DATA_DIR.glob('*.csv'))
    # Check if cached data exists for both the selected timeframe and reference timeframe
    all_cached = all(
        (utils.CACHE_DIR / utils.get_cache_filename(filename, selected_tf_code)).exists() and
        (utils.CACHE_DIR / utils.get_cache_filename(filename, selected_reference_tf_code)).exists()
        for filename in selected_files
    )

    with st.spinner("Loading data..."):
        # Load and combine reference timeframe data
        reference_combined = None
        for filename in selected_files:
            reference_cache_path = utils.CACHE_DIR / utils.get_cache_filename(filename, selected_reference_tf_code)
            if reference_cache_path.exists():
                reference_data = pd.read_pickle(reference_cache_path)
                if reference_combined is None:
                    reference_combined = reference_data
                else:
                    reference_combined = pd.concat([reference_combined, reference_data])

        # Load and combine selected timeframe data
        h1_combined = None
        for filename in selected_files:
            h1_cache_path = utils.CACHE_DIR / utils.get_cache_filename(filename, selected_tf_code)
            if h1_cache_path.exists():
                h1_data = pd.read_pickle(h1_cache_path)
                if h1_combined is None:
                    h1_combined = h1_data
                else:
                    h1_combined = pd.concat([h1_combined, h1_data])

    # Check if data was loaded successfully
    if reference_combined is None:
        st.error("No reference timeframe data available. Please check your cache files.")
        return
        
    if h1_combined is None:
        st.error("No data available for the selected timeframe. Please check your cache files.")
        return

    # Sort the data
    reference_combined = reference_combined.sort_index()
    h1_combined = h1_combined.sort_index()
    
    results = analysis.analyze_candle_batch(h1_combined, reference_combined, selected_reference_tf_code, 
                                          tp_percent, sl_percent, enable_end_of_tf_restriction,
                                          enable_reverse_calculation)
    
    # Filter results based on selected days
    if selected_days:
        results = results[results['day_of_week'].isin(selected_days)]

    st.session_state.results = results

    if len(results) == 0:
        st.error("No results were generated. Please check if your data contains valid price movements.")
    else:
        results['tf_datetime'] = pd.to_datetime(results['tf_datetime'])
        for direction in ["up", "down"]:
            mask = (results['reference_direction'] == direction)
            direction_data = results[mask]
            if len(direction_data) > 0:
                success_count = np.sum(direction_data['hit_target_first'])
                total_count = np.sum(direction_data['hit_target_first'] | direction_data['hit_stoploss_first'])
                probability = (success_count / total_count * 100) if total_count > 0 else 0
                results.loc[mask, 'probability'] = probability

        display.display_analysis_results(results, selected_tf_code)
if prepare_data:
    data_preparation.prepare_and_cache_data(selected_tf_code, selected_reference_tf_code)

if start_analysis:
    run_analysis(selected_tf_code, selected_reference_tf_code, tp_percent, sl_percent, enable_end_of_tf_restriction, enable_reverse_calculation, selected_days)

if filter_best_candles:
    data_preparation.filter_best_candles(selected_reference_tf, selected_tf,selected_tf_code)
