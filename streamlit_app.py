import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import io
import os
import pickle
import swifter  # For parallel processing
import pytz
from datetime import datetime

st.set_page_config(layout="wide", page_title="Financial Data Analysis")

st.title("Financial Price Movement Probability Analysis")
st.markdown("""
This app analyzes financial futures data to determine probability of price movements.
For each H1 candle, it checks if the first M5 candle closes above the H1 opening price,
then calculates the probability of a 0.1% move in the direction of the M5 close before
a 0.2% move in the opposite direction.
""")

# Parameters
tp_percent = st.sidebar.number_input("Take Profit %", value=0.1, step=0.05, format="%.2f")
sl_percent = st.sidebar.number_input("Stop Loss %", value=0.2, step=0.05, format="%.2f")

# Define directories
DATA_DIR = Path(__file__).parent / 'data'
CACHE_DIR = Path(__file__).parent / 'cached_data'

# Create cache directory if it doesn't exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Get available CSV files
available_files = list(DATA_DIR.glob('*.csv'))

# Let user select files from a dropdown
selected_files = st.sidebar.multiselect(
    "Select CSV files to analyze",
    options=[file.name for file in available_files]
)

# Timeframe selection for resampling
resampling_options = ["5 Minutes (M5)", "15 Minutes (M15)", "30 Minutes (M30)", "1 Hour (H1)", "4 Hours (H4)"]
resample_mapping = {
    "5 Minutes (M5)": "5T",
    "15 Minutes (M15)": "15T",
    "30 Minutes (M30)": "30T",
    "1 Hour (H1)": "1H",
    "4 Hours (H4)": "4H"
}

selected_tf = st.sidebar.selectbox(
    "Select larger timeframe",
    options=resampling_options,
    index=3  # Default to H1
)

selected_tf_code = resample_mapping[selected_tf]

# Function to get cache filename for a specific file and timeframe
def get_cache_filename(csv_filename, timeframe_code):
    base_name = os.path.splitext(csv_filename)[0]
    return f"{base_name}_{timeframe_code.replace(':', '_')}.pkl"

# Function to get M5 cache filename for a specific file
def get_m5_cache_filename(csv_filename):
    base_name = os.path.splitext(csv_filename)[0]
    return f"{base_name}_5T.pkl"

def convert_to_ny_local_time(df, timestamp_column='datetime'):
    """
    Convert timestamps in the DataFrame from UTC+1 to New York local time.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the timestamps.
        timestamp_column (str): The name of the timestamp column.
    
    Returns:
        pd.DataFrame: The DataFrame with timestamps converted to NY local time.
    """
    # Ensure the timestamp column is in datetime format
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    
    # Convert from UTC+1 to UTC (subtract 1 hour)
    df[timestamp_column] = df[timestamp_column] - pd.Timedelta(hours=1)
    
    # Convert from UTC to New York local time (considering daylight savings)
    utc_tz = pytz.UTC
    ny_tz = pytz.timezone('America/New_York')
    
    # Localize the timestamps to UTC
    df[timestamp_column] = df[timestamp_column].dt.tz_localize(utc_tz)
    
    # Convert to New York local time
    df[timestamp_column] = df[timestamp_column].dt.tz_convert(ny_tz)
    
    # Remove timezone information for easier handling (optional)
    df[timestamp_column] = df[timestamp_column].dt.tz_localize(None)
    
    return df

# Data preparation section remains the same...
st.sidebar.markdown("## Data Preparation")
prepare_data = st.sidebar.button("Prepare & Cache Data")

if prepare_data:
    if not selected_files:
        st.sidebar.error("No files selected. Please select at least one CSV file.")
    else:
        # Create a progress bar
        prep_progress = st.sidebar.progress(0)
        prep_status = st.sidebar.empty()
        
        for i, filename in enumerate(selected_files):
            prep_status.text(f"Processing {filename}... ({i+1}/{len(selected_files)})")
            
            try:
                file_path = DATA_DIR / filename
                
                # Read the raw CSV
                df = pd.read_csv(file_path, delimiter=';')
                df.columns = df.columns.str.strip()  # Clean column names
                
                # Expected columns for this specific format
                expected_columns = ['date', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'year']
                
                # Check if all expected columns exist (case-insensitive)
                missing_columns = [col for col in expected_columns if not any(existing_col.lower() == col.lower() for existing_col in df.columns)]
                
                if missing_columns:
                    st.sidebar.warning(f"File {filename}: Missing columns: {', '.join(missing_columns)}. Skipping.")
                    continue
                
                # Map actual column names to expected column names (case-insensitive)
                column_mapping = {}
                for expected_col in expected_columns:
                    for actual_col in df.columns:
                        if actual_col.lower() == expected_col.lower():
                            column_mapping[expected_col] = actual_col
                
                # Create a copy with standardized column names
                df_processed = df.rename(columns={column_mapping[col]: col for col in expected_columns if col in column_mapping})
                
                # Combine date and timestamp to create datetime
                df_processed['datetime'] = pd.to_datetime(df_processed['date'] + ' ' + df_processed['timestamp'])
                
                df_processed = convert_to_ny_local_time(df_processed, timestamp_column='datetime')
                
                # Ensure numeric columns are numeric
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                
                # Sort by datetime and drop missing values
                df_processed = df_processed.sort_values('datetime')
                df_processed = df_processed.dropna(subset=['datetime', 'open', 'high', 'low', 'close'])
                
                # Set datetime as index for resampling
                df_processed = df_processed.set_index('datetime')
                
                # Create M5 data and save it
                m5_data = df_processed.resample('5T').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                m5_cache_path = CACHE_DIR / get_m5_cache_filename(filename)
                m5_data.to_pickle(m5_cache_path)
                
                # Create selected timeframe data and save it
                selected_tf_data = df_processed.resample(selected_tf_code).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                
                selected_tf_cache_path = CACHE_DIR / get_cache_filename(filename, selected_tf_code)
                selected_tf_data.to_pickle(selected_tf_cache_path)
                
            except Exception as e:
                st.sidebar.error(f"Error processing {filename}: {str(e)}")
            
            # Update progress
            prep_progress.progress((i + 1) / len(selected_files))
        
        prep_status.text("Data preparation complete!")
        st.sidebar.success(f"Processed and cached data for {len(selected_files)} files")

# Optimized analysis function that works on batches of data
def analyze_candle_batch(batch_df, m5_combined, tp_percent, sl_percent):
    batch_results = []
    
    for tf_time, tf_row in batch_df.iterrows():
        # Calculate end time based on timeframe
        if '1H' in tf_time.freq.name:
            tf_end_time = tf_time + pd.Timedelta(hours=1) - pd.Timedelta(seconds=1)
        elif '4H' in tf_time.freq.name:
            tf_end_time = tf_time + pd.Timedelta(hours=4) - pd.Timedelta(seconds=1)
        elif '5T' in tf_time.freq.name:
            tf_end_time = tf_time + pd.Timedelta(minutes=5) - pd.Timedelta(seconds=1)
        elif '15T' in tf_time.freq.name:
            tf_end_time = tf_time + pd.Timedelta(minutes=15) - pd.Timedelta(seconds=1)
        elif '30T' in tf_time.freq.name:
            tf_end_time = tf_time + pd.Timedelta(minutes=30) - pd.Timedelta(seconds=1)
        else:
            # Default to 1 hour if frequency is not detected
            tf_end_time = tf_time + pd.Timedelta(hours=1) - pd.Timedelta(seconds=1)
            
        # Get M5 candles within this tf candle - use .loc with slice for performance
        m5_candles_in_tf = m5_combined.loc[tf_time:tf_end_time]
        
        if len(m5_candles_in_tf) > 0:
            first_m5 = m5_candles_in_tf.iloc[0]
            
            # Check direction
            m5_direction = "up" if first_m5['close'] > tf_row['open'] else "down"
            
            # Calculate target and stop levels once
            if m5_direction == "up":
                target_level = first_m5['close'] * (1 + tp_percent/100)
                stop_level = first_m5['close'] * (1 - sl_percent/100)
            else:
                target_level = first_m5['close'] * (1 - tp_percent/100)
                stop_level = first_m5['close'] * (1 + sl_percent/100)
            
            # Get next M5 candles after the first one
            next_candles = m5_candles_in_tf.iloc[1:]
            
            # Optimization: Use vectorized operations instead of loop when possible
            if len(next_candles) > 0:
                if m5_direction == "up":
                    # Check if any high >= target or any low <= stop
                    hits_target = next_candles['high'] >= target_level
                    hits_stop = next_candles['low'] <= stop_level
                else:
                    # Check if any low <= target or any high >= stop
                    hits_target = next_candles['low'] <= target_level
                    hits_stop = next_candles['high'] >= stop_level
                
                # Find the first occurrence of either condition
                hit_target_first = False
                hit_stoploss_first = False
                
                if hits_target.any() or hits_stop.any():
                    # Find first index where either condition is true
                    first_target_idx = hits_target.idxmax() if hits_target.any() else None
                    first_stop_idx = hits_stop.idxmax() if hits_stop.any() else None
                    
                    # Determine which came first
                    if first_target_idx is not None and first_stop_idx is not None:
                        target_time = next_candles.index.get_loc(first_target_idx)
                        stop_time = next_candles.index.get_loc(first_stop_idx)
                        hit_target_first = target_time <= stop_time
                        hit_stoploss_first = stop_time < target_time
                    elif first_target_idx is not None:
                        hit_target_first = True
                    elif first_stop_idx is not None:
                        hit_stoploss_first = True
            
            # Add to results
            batch_results.append({
                'tf_datetime': tf_time,
                'tf_open': tf_row['open'],
                'first_m5_close': first_m5['close'],
                'm5_direction': m5_direction,
                'hit_target_first': hit_target_first,
                'hit_stoploss_first': hit_stoploss_first,
                'probability': None  # Will calculate later
            })
    
    return pd.DataFrame(batch_results)

# ------------------ OPTIMIZED ANALYSIS SECTION ------------------
st.sidebar.markdown("## Analysis")
start_analysis = st.sidebar.button("Start Analysis")

# Function to preprocess and return the datetime timedeltas based on timeframe
def get_timedelta_for_timeframe(tf_code):
    if tf_code == "1H":
        return pd.Timedelta(hours=1)
    elif tf_code == "4H":
        return pd.Timedelta(hours=4)
    elif tf_code == "5T":
        return pd.Timedelta(minutes=5)
    elif tf_code == "15T":
        return pd.Timedelta(minutes=15)
    elif tf_code == "30T":
        return pd.Timedelta(minutes=30)
    else:
        return pd.Timedelta(hours=1)  # Default to 1 hour

# Optimized function to identify hourly M5 candles directly from M5 data
def identify_first_m5_of_period(m5_data, tf_code):
    """
    Identifies the first M5 candle of each higher timeframe period directly from M5 data.
    This eliminates the need to load and process higher timeframe data separately.
    
    Parameters:
    m5_data (DataFrame): DataFrame containing all M5 candles
    tf_code (str): Timeframe code (e.g., '1H', '4H', etc.)
    
    Returns:
    DataFrame: DataFrame with indices of the first M5 candle of each higher timeframe period
    """
    # Ensure the index is a DatetimeIndex
    if not isinstance(m5_data.index, pd.DatetimeIndex):
        m5_data.index = pd.to_datetime(m5_data.index)
    
    # Create a helper column for the timeframe start
    if tf_code == "1H":
        # Get the first M5 candle of each hour
        m5_data['tf_start'] = m5_data.index.floor('H')
    elif tf_code == "4H":
        # Get the first M5 candle of each 4-hour period
        m5_data['tf_start'] = m5_data.index.floor('4H')
    elif tf_code == "15T":
        # Get the first M5 candle of each 15-minute period
        m5_data['tf_start'] = m5_data.index.floor('15T')
    elif tf_code == "30T":
        # Get the first M5 candle of each 30-minute period
        m5_data['tf_start'] = m5_data.index.floor('30T')
    else:
        # Default to 1H if not specified
        m5_data['tf_start'] = m5_data.index.floor('H')
    
    # Group by the timeframe start and get the first M5 candle of each group
    first_m5_indices = m5_data.groupby('tf_start').apply(lambda x: x.index[0]).tolist()
    
    # Get these specific M5 candles
    first_m5_candles = m5_data.loc[first_m5_indices].copy()
    
    # Add the tf_start as a column for reference
    first_m5_candles['tf_datetime'] = first_m5_candles['tf_start']
    
    # Clean up
    first_m5_candles.drop('tf_start', axis=1, inplace=True)
    m5_data.drop('tf_start', axis=1, inplace=True)
    
    return first_m5_candles

# Optimized function to analyze price movements directly from M5 data
def analyze_price_movements(first_m5_candles, m5_data, tp_percent, sl_percent):
    """
    Analyzes price movements to determine the probability of hitting target profit before stop loss.
    
    Parameters:
    first_m5_candles (DataFrame): DataFrame containing the first M5 candle of each higher timeframe period
    m5_data (DataFrame): DataFrame containing all M5 candles
    tp_percent (float): Take profit percentage
    sl_percent (float): Stop loss percentage
    
    Returns:
    DataFrame: Results of the analysis
    """
    results = []
    
    # Create a progress bar
    total_candles = len(first_m5_candles)
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    # Process in batches for better performance
    batch_size = 100
    batches = np.array_split(first_m5_candles.index, max(1, total_candles // batch_size))
    
    for i, batch_indices in enumerate(batches):
        # Update progress
        progress_bar.progress((i + 1) / len(batches))
        progress_text.text(f"Processing batch {i+1} of {len(batches)}")
        
        batch_candles = first_m5_candles.loc[batch_indices]
        
        for idx, row in batch_candles.iterrows():
            # The opening price of the timeframe is the same as the first M5 opening price
            tf_open = row['open']
            m5_close = row['close']
            
            # Check if M5 closes above or below the opening price
            m5_direction = "up" if m5_close > tf_open else "down"
            
            # Calculate target and stop levels
            if m5_direction == "up":
                target_level = m5_close * (1 + tp_percent/100)
                stop_level = m5_close * (1 - sl_percent/100)
            else:
                target_level = m5_close * (1 - tp_percent/100)
                stop_level = m5_close * (1 + sl_percent/100)
            
            # Get the end time of the current timeframe period
            if row['tf_datetime'].hour != idx.hour and 'tf_datetime' in row:
                # Use the provided tf_datetime if available and different
                tf_end_time = row['tf_datetime'] + get_timedelta_for_timeframe(selected_tf_code) - pd.Timedelta(seconds=1)
            else:
                # Calculate based on the M5 candle time
                tf_time = idx
                tf_end_time = tf_time.floor('H') + pd.Timedelta(hours=1) - pd.Timedelta(seconds=1)
                if selected_tf_code == "4H":
                    tf_end_time = tf_time.floor('4H') + pd.Timedelta(hours=4) - pd.Timedelta(seconds=1)
                elif selected_tf_code == "15T":
                    tf_end_time = tf_time.floor('15T') + pd.Timedelta(minutes=15) - pd.Timedelta(seconds=1)
                elif selected_tf_code == "30T":
                    tf_end_time = tf_time.floor('30T') + pd.Timedelta(minutes=30) - pd.Timedelta(seconds=1)
            
            # Get M5 candles after this first one until the end of the timeframe
            next_candles = m5_data.loc[idx:tf_end_time].iloc[1:]
            
            hit_target_first = False
            hit_stoploss_first = False
            
            if len(next_candles) > 0:
                if m5_direction == "up":
                    hits_target = next_candles['high'] >= target_level
                    hits_stop = next_candles['low'] <= stop_level
                else:
                    hits_target = next_candles['low'] <= target_level
                    hits_stop = next_candles['high'] >= stop_level
                
                if hits_target.any() or hits_stop.any():
                    # Find first index where either condition is true
                    first_target_idx = hits_target.idxmax() if hits_target.any() else None
                    first_stop_idx = hits_stop.idxmax() if hits_stop.any() else None
                    
                    # Determine which came first
                    if first_target_idx is not None and first_stop_idx is not None:
                        target_time = next_candles.index.get_loc(first_target_idx)
                        stop_time = next_candles.index.get_loc(first_stop_idx)
                        hit_target_first = target_time <= stop_time
                        hit_stoploss_first = stop_time < target_time
                    elif first_target_idx is not None:
                        hit_target_first = True
                    elif first_stop_idx is not None:
                        hit_stoploss_first = True
            
            # Add to results
            results.append({
                'tf_datetime': row['tf_datetime'] if 'tf_datetime' in row else idx.floor('H'),
                'tf_open': tf_open,
                'first_m5_close': m5_close,
                'm5_direction': m5_direction,
                'hit_target_first': hit_target_first,
                'hit_stoploss_first': hit_stoploss_first,
                'probability': None  # Will calculate later
            })
    
    # Clear progress indicators
    progress_bar.empty()
    progress_text.empty()
    
    return pd.DataFrame(results)

# Only proceed with analysis if button is pressed
if start_analysis:
    if not selected_files:
        st.sidebar.error("No files selected. Please select at least one CSV file.")
    else:
        # Check if cached data exists for M5 files
        all_cached = True
        for filename in selected_files:
            m5_cache_path = CACHE_DIR / get_m5_cache_filename(filename)
            if not m5_cache_path.exists():
                all_cached = False
                break
        
        if not all_cached:
            st.warning("Some files have not been prepared. Please click 'Prepare & Cache Data' first.")
        else:
            with st.spinner("Loading M5 data and identifying first M5 candles of each period..."):
                # Load and combine all M5 data
                m5_combined = None
                
                for filename in selected_files:                   
                    m5_cache_path = CACHE_DIR / get_m5_cache_filename(filename)
                    m5_data = pd.read_pickle(m5_cache_path)
                    
                    if m5_combined is None:
                        m5_combined = m5_data
                    else:
                        m5_combined = pd.concat([m5_combined, m5_data])
                
                # Sort by datetime
                m5_combined = m5_combined.sort_index()
                
                # Identify the first M5 candle of each higher timeframe period
                first_m5_candles = identify_first_m5_of_period(m5_combined, selected_tf_code)
                
            # Show info about the data
            st.info(f"Loaded {len(first_m5_candles)} first M5 candles of {selected_tf} periods and {len(m5_combined)} total 5-minute candles")
            
            # Analyze price movements
            results = analyze_price_movements(first_m5_candles, m5_combined, tp_percent, sl_percent)
            
            # Store results in session state
            st.session_state.results = results
            
            # Clear progress indicators
            #progress_bar.empty()
            #progress_text.empty()
            
            if len(results) == 0:
                st.error("No results were generated. Please check if your data contains valid price movements.")
            else:
                # Convert strings to datetime objects if needed
                if not isinstance(results['tf_datetime'].iloc[0], pd.Timestamp):
                    results['tf_datetime'] = pd.to_datetime(results['tf_datetime'])
                
                # Calculate probabilities by direction (vectorized operations)
                for direction in ["up", "down"]:
                    mask = (results['m5_direction'] == direction)
                    direction_data = results[mask]
                    
                    if len(direction_data) > 0:
                        # Use numpy for faster calculations
                        success_count = np.sum(direction_data['hit_target_first'])
                        total_count = np.sum(direction_data['hit_target_first'] | direction_data['hit_stoploss_first'])
                        
                        probability = (success_count / total_count * 100) if total_count > 0 else 0
                        
                        # Update all rows at once using loc
                        results.loc[mask, 'probability'] = probability
                
                # Display results
                st.subheader("Analysis Results")
                
                # Display summary statistics
                st.markdown("### Summary Statistics")

                col1, col2 = st.columns(2)

                with col1:
                    up_data = results[results['m5_direction'] == "up"]
                    up_success = np.sum(up_data['hit_target_first'])
                    up_total = np.sum(up_data['hit_target_first'] | up_data['hit_stoploss_first'])
                    up_probability = (up_success / up_total * 100) if up_total > 0 else 0
                    
                    st.metric(
                        label=f"When first M5 closes ABOVE {selected_tf} open",
                        value=f"{up_probability:.2f}%",
                        delta=f"{up_success}/{up_total} cases"
                    )

                with col2:
                    down_data = results[results['m5_direction'] == "down"]
                    down_success = np.sum(down_data['hit_target_first'])
                    down_total = np.sum(down_data['hit_target_first'] | down_data['hit_stoploss_first'])
                    down_probability = (down_success / down_total * 100) if down_total > 0 else 0
                    
                    st.metric(
                        label=f"When first M5 closes BELOW {selected_tf} open",
                        value=f"{down_probability:.2f}%",
                        delta=f"{down_success}/{down_total} cases"
                    )
                
                # Display detailed results table - with pagination for better performance
                st.markdown("### Detailed Analysis by Timeframe Candle")
                
                # Filter options                        
                direction_filter = st.multiselect(
                    "Filter by Direction",
                    options=["up", "down"],
                    default=["up", "down"]
                )
                
                # Apply filters - use efficient boolean indexing
                filtered_results = results[results['m5_direction'].isin(direction_filter)]
                
                # For better performance with large datasets, use pagination
                page_size = 1000
                total_pages = (len(filtered_results) + page_size - 1) // page_size
                
                if total_pages > 1:
                    page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
                    start_idx = (page_number - 1) * page_size
                    end_idx = min(start_idx + page_size, len(filtered_results))
                    display_results = filtered_results.iloc[start_idx:end_idx]
                    st.info(f"Showing results {start_idx+1} to {end_idx} of {len(filtered_results)}")
                else:
                    display_results = filtered_results
                
                # Display only essential columns for better performance
                display_cols = ['tf_datetime', 'tf_open', 'first_m5_close', 'm5_direction', 'hit_target_first', 'probability']
                st.dataframe(display_results[display_cols])
                
                # Create visualization
                st.markdown("### Visualization")
                
                # Optimize hour of day analysis using efficient groupby operations
                results['hour'] = results['tf_datetime'].dt.hour
                
                # Faster aggregation by pre-computing the metrics for each group
                hourly_data = []
                
                for hour in range(24):  # 24 hours in a day
                    for direction in ['up', 'down']:
                        hour_direction_mask = (results['hour'] == hour) & (results['m5_direction'] == direction)
                        if np.any(hour_direction_mask):
                            success = np.sum(results.loc[hour_direction_mask, 'hit_target_first'])
                            total = np.sum(results.loc[hour_direction_mask, 'hit_target_first'] | 
                                          results.loc[hour_direction_mask, 'hit_stoploss_first'])
                            if total > 0:
                                probability = (success / total) * 100
                                hourly_data.append({
                                    'hour': hour,
                                    'direction': direction,
                                    'success': success,
                                    'total': total,
                                    'probability': probability
                                })
                
                hourly_df = pd.DataFrame(hourly_data)
                
                # Only create visualization if we have data
                if not hourly_df.empty:
                    # Separate by direction for plotting
                    up_df = hourly_df[hourly_df['direction'] == 'up']
                    down_df = hourly_df[hourly_df['direction'] == 'down']
                    
                    fig = go.Figure()
                    
                    # Add "up" direction bars if they exist
                    if not up_df.empty:
                        fig.add_trace(go.Bar(
                            x=up_df['hour'],
                            y=up_df['probability'],
                            name='Up Direction',
                            marker_color='green',
                            opacity=0.7,
                            text=up_df['success'].astype(str) + '/' + up_df['total'].astype(str),
                            hoverinfo='text'
                        ))
                    
                    # Add "down" direction bars if they exist
                    if not down_df.empty:
                        fig.add_trace(go.Bar(
                            x=down_df['hour'],
                            y=down_df['probability'],
                            name='Down Direction',
                            marker_color='red',
                            opacity=0.7,
                            text=down_df['success'].astype(str) + '/' + down_df['total'].astype(str),
                            hoverinfo='text'
                        ))
                    
                    fig.update_layout(
                        title='Success Probability by Hour of Day',
                        xaxis_title='Hour of Day',
                        yaxis_title='Probability (%)',
                        barmode='group',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough data for hourly visualization.")
                
                # Add a cache for the analysis results
                if st.button("Cache These Results for Faster Future Access"):
                    results_cache_path = CACHE_DIR / f"analysis_results_{selected_tf_code}.pkl"
                    results.to_pickle(results_cache_path)
                    st.success(f"Results cached to {results_cache_path}")
                
                # Export option - optimize for large datasets
                st.subheader("Export Results")
                
                export_options = st.radio(
                    "Export Options", 
                    ["Download Complete Results", "Download Filtered Results"]
                )
                
                if export_options == "Download Complete Results":
                    csv_data = results.to_csv(index=False)
                    file_name = f"price_movement_analysis_{selected_tf_code}_complete.csv"
                else:
                    csv_data = filtered_results.to_csv(index=False)
                    file_name = f"price_movement_analysis_{selected_tf_code}_filtered.csv" 
                
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_data,
                    file_name=file_name,
                    mime="text/csv"
                )    
        # Store results in session state
        st.session_state.results = results

# Best H1 Candles Analysis
st.sidebar.markdown("## Best H1 Candle Analysis")
filter_best_candles = st.sidebar.button("Filter by Best H1 Candles")

if filter_best_candles:
    # Check if results are stored in session state
    if 'results' not in st.session_state:
        st.warning("Please run the analysis first before filtering for best candles.")
    else:
        # Retrieve results from session state
        results = st.session_state.results
        
        # Create a new section for best candles analysis
        st.subheader("Best Hourly Candles Analysis (Above 60% Probability)")
        
        # Group by hour and direction to find best performing hours
        results['hour'] = results['tf_datetime'].dt.hour
        
        # Calculate performance metrics for each hour and direction
        best_hours_data = []
        
        for hour in range(24):
            for direction in ['up', 'down']:
                mask = (results['hour'] == hour) & (results['m5_direction'] == direction)
                hour_direction_data = results[mask]
                
                if len(hour_direction_data) > 0:
                    success = np.sum(hour_direction_data['hit_target_first'])
                    total = np.sum(hour_direction_data['hit_target_first'] | hour_direction_data['hit_stoploss_first'])
                    
                    if total >= 30:  # Minimum sample size for statistical significance
                        probability = (success / total) * 100
                        
                        best_hours_data.append({
                            'hour': hour,
                            'direction': direction,
                            'success': success,
                            'total': total,
                            'probability': probability,
                            'sample_size': len(hour_direction_data)
                        })
        
        # Convert to DataFrame for analysis
        best_hours_df = pd.DataFrame(best_hours_data)
        
        # Filter for hours with probability > 60%
        high_prob_hours = best_hours_df[best_hours_df['probability'] > 60].sort_values('probability', ascending=False)
        
        if len(high_prob_hours) > 0:
            # Display the high probability hours
            st.markdown("### Hourly Candles with > 60% Probability")
            
            # Format the display of high probability hours
            display_cols = ['hour', 'direction', 'probability', 'success', 'total', 'sample_size']
            formatted_high_prob = high_prob_hours[display_cols].copy()
            formatted_high_prob['probability'] = formatted_high_prob['probability'].round(2).astype(str) + '%'
            formatted_high_prob['success_rate'] = formatted_high_prob['success'].astype(str) + '/' + formatted_high_prob['total'].astype(str)
            
            st.dataframe(formatted_high_prob[['hour', 'direction', 'probability', 'success_rate', 'sample_size']])
            
            # Create a filter criteria for the original results based on these high probability hours
            filtered_indices = []
            
            for _, row in high_prob_hours.iterrows():
                hour = row['hour']
                direction = row['direction']
                indices = results[(results['hour'] == hour) & (results['m5_direction'] == direction)].index
                filtered_indices.extend(indices)
            
            # Filter the original results
            high_prob_results = results.loc[filtered_indices]
            
            # Recalculate overall metrics using only the high probability candles
            st.markdown("### Recalculated Metrics Using Only High Probability Hours")
            
            col1, col2 = st.columns(2)
            
            with col1:
                filtered_up_data = high_prob_results[high_prob_results['m5_direction'] == "up"]
                filtered_up_success = np.sum(filtered_up_data['hit_target_first'])
                filtered_up_total = np.sum(filtered_up_data['hit_target_first'] | filtered_up_data['hit_stoploss_first'])
                filtered_up_probability = (filtered_up_success / filtered_up_total * 100) if filtered_up_total > 0 else 0
                
                st.metric(
                    label=f"When first M5 closes ABOVE {selected_tf} open (filtered)",
                    value=f"{filtered_up_probability:.2f}%",
                    delta=f"{filtered_up_success}/{filtered_up_total} cases"
                )
            
            with col2:
                filtered_down_data = high_prob_results[high_prob_results['m5_direction'] == "down"]
                filtered_down_success = np.sum(filtered_down_data['hit_target_first'])
                filtered_down_total = np.sum(filtered_down_data['hit_target_first'] | filtered_down_data['hit_stoploss_first'])
                filtered_down_probability = (filtered_down_success / filtered_down_total * 100) if filtered_down_total > 0 else 0
                
                st.metric(
                    label=f"When first M5 closes BELOW {selected_tf} open (filtered)",
                    value=f"{filtered_down_probability:.2f}%",
                    delta=f"{filtered_down_success}/{filtered_down_total} cases"
                )
            
            # Overall combined metric
            st.metric(
                label="Overall Success Rate (High Probability Hours Only)",
                value=f"{((filtered_up_success + filtered_down_success) / (filtered_up_total + filtered_down_total) * 100):.2f}%",
                delta=f"{filtered_up_success + filtered_down_success}/{filtered_up_total + filtered_down_total} cases"
            )
            
            # Visualization of filtered results
            st.markdown("### Visualization of High Probability Hours")
            
            fig = go.Figure()
            
            for direction in ['up', 'down']:
                direction_data = high_prob_hours[high_prob_hours['direction'] == direction]
                
                if not direction_data.empty:
                    color = 'green' if direction == 'up' else 'red'
                    
                    fig.add_trace(go.Bar(
                        x=direction_data['hour'],
                        y=direction_data['probability'],
                        name=f'{direction.capitalize()} Direction',
                        marker_color=color,
                        opacity=0.7,
                        text=direction_data['success'].astype(str) + '/' + direction_data['total'].astype(str),
                        hoverinfo='text'
                    ))
            
            fig.update_layout(
                title='High Probability Hours (Above 60%)',
                xaxis_title='Hour of Day',
                yaxis_title='Probability (%)',
                barmode='group',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Option to download the filtered results
            filtered_csv_data = high_prob_results.to_csv(index=False)
            st.download_button(
                label="Download High Probability Results as CSV",
                data=filtered_csv_data,
                file_name=f"high_probability_{selected_tf_code}_results.csv",
                mime="text/csv"
            )
        else:
            st.warning("No hourly candles found with probability above 60%. Try lowering the threshold or analyzing more data.")