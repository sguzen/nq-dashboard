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

# Constants
DATA_DIR = Path(__file__).parent / 'data'
CACHE_DIR = Path(__file__).parent / 'cached_data'

# Create cache directory if it doesn't exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Streamlit configuration
st.set_page_config(layout="wide", page_title="Financial Data Analysis")
st.title("Financial Price Movement Probability Analysis")
st.markdown("""
This app analyzes financial futures data to determine probability of price movements.
For each H1 candle, it checks if the first M5 candle closes above the H1 opening price,
then calculates the probability of a 0.1% move in the direction of the M5 close before
a XX% move in the opposite direction.
""")

# Sidebar parameters
tp_percent = st.sidebar.number_input("Take Profit %", value=0.1, step=0.05, format="%.2f")
sl_percent = st.sidebar.number_input("Stop Loss %", value=0.2, step=0.05, format="%.2f")

# Reference timeframe selection
reference_timeframe_options = ["1 Minute (M1)", "5 Minutes (M5)", "15 Minutes (M15)", "30 Minutes (M30)"]
reference_timeframe_mapping = {
    "1 Minute (M1)": "1T",
    "5 Minutes (M5)": "5T",
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

# Toggle switch for end-of-timeframe restriction
enable_end_of_tf_restriction = st.sidebar.checkbox(
    "Classify scenarios as losers if neither target nor stop level is hit by the end of the larger timeframe",
    value=True  # Default to enabled
)

# Get available CSV files
available_files = list(DATA_DIR.glob('*.csv'))
selected_files = st.sidebar.multiselect(
    "Select CSV files to analyze",
    options=[file.name for file in available_files]
)

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

# Function to prepare and cache data
def prepare_and_cache_data(selected_files, selected_tf_code, selected_reference_tf_code):
    if not selected_files:
        st.sidebar.error("No files selected. Please select at least one CSV file.")
        return

    prep_progress = st.sidebar.progress(0)
    prep_status = st.sidebar.empty()

    for i, filename in enumerate(selected_files):
        prep_status.text(f"Processing {filename}... ({i+1}/{len(selected_files)})")
        try:
            file_path = DATA_DIR / filename
            df = pd.read_csv(file_path, delimiter=';')
            df.columns = df.columns.str.strip()

            expected_columns = ['date', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'year']
            missing_columns = [col for col in expected_columns if not any(existing_col.lower() == col.lower() for existing_col in df.columns)]

            if missing_columns:
                st.sidebar.warning(f"File {filename}: Missing columns: {', '.join(missing_columns)}. Skipping.")
                continue

            column_mapping = {expected_col: actual_col for expected_col in expected_columns for actual_col in df.columns if actual_col.lower() == expected_col.lower()}
            df_processed = df.rename(columns={column_mapping[col]: col for col in expected_columns if col in column_mapping})
            df_processed['datetime'] = pd.to_datetime(df_processed['date'] + ' ' + df_processed['timestamp'])
            df_processed = convert_to_ny_local_time(df_processed, timestamp_column='datetime')

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

            df_processed = df_processed.sort_values('datetime')
            df_processed = df_processed.dropna(subset=['datetime', 'open', 'high', 'low', 'close'])
            df_processed = df_processed.set_index('datetime')

            # Cache M1 data
            m1_data = df_processed.resample('1T').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            m1_cache_path = CACHE_DIR / get_cache_filename(filename, "1T")
            m1_data.to_pickle(m1_cache_path)

            # Cache reference timeframe data (optional, if needed for other purposes)
            reference_data = df_processed.resample(selected_reference_tf_code).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            reference_cache_path = CACHE_DIR / get_cache_filename(filename, selected_reference_tf_code)
            reference_data.to_pickle(reference_cache_path)

            # Cache selected timeframe data
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

        prep_progress.progress((i + 1) / len(selected_files))

    prep_status.text("Data preparation complete!")
    st.sidebar.success(f"Processed and cached data for {len(selected_files)} files")

# Function to analyze candle batch
def analyze_candle_batch(h1_data, reference_data, tp_percent, sl_percent, enable_end_of_tf_restriction):
    batch_results = []

    for tf_time, tf_row in h1_data.iterrows():
        tf_end_time = tf_time + pd.Timedelta(hours=1) - pd.Timedelta(seconds=1)
        reference_candles_in_tf = reference_data.loc[tf_time:tf_end_time]

        if len(reference_candles_in_tf) > 0:
            first_reference_candle = reference_candles_in_tf.iloc[0]
            reference_direction = "up" if first_reference_candle['close'] > first_reference_candle['open'] else "down"

            if reference_direction == "up":
                target_level = first_reference_candle['close'] * (1 + tp_percent / 100)
                stop_level = first_reference_candle['close'] * (1 - sl_percent / 100)
            else:
                target_level = first_reference_candle['close'] * (1 - tp_percent / 100)
                stop_level = first_reference_candle['close'] * (1 + sl_percent / 100)

            next_candles = reference_candles_in_tf.iloc[1:]
            hit_target = False
            hit_stop = False

            for _, candle in next_candles.iterrows():
                if reference_direction == "up":
                    if candle['high'] >= target_level:
                        hit_target = True
                        break
                    if candle['low'] <= stop_level:
                        hit_stop = True
                        break
                else:
                    if candle['low'] <= target_level:
                        hit_target = True
                        break
                    if candle['high'] >= stop_level:
                        hit_stop = True
                        break

            # Handle end-of-timeframe restriction based on the toggle switch
            if enable_end_of_tf_restriction and not hit_target and not hit_stop:
                hit_stop = True  # Mark as a loser if neither target nor stop level is hit by the end of the larger timeframe

            hit_target_first = hit_target
            hit_stoploss_first = hit_stop

            batch_results.append({
                'tf_datetime': tf_time,
                'tf_open': tf_row['open'],
                'first_reference_close': first_reference_candle['close'],
                'reference_direction': reference_direction,
                'hit_target_first': hit_target_first,
                'hit_stoploss_first': hit_stoploss_first,
                'probability': None
            })

    return pd.DataFrame(batch_results)

# Function to load and combine cached data
def load_and_combine_cached_data(selected_files, selected_tf_code):
    m5_combined = None
    h1_combined = None

    for filename in selected_files:
        m5_cache_path = CACHE_DIR / get_cache_filename(filename, "5T")
        h1_cache_path = CACHE_DIR / get_cache_filename(filename, selected_tf_code)

        if m5_cache_path.exists():
            m5_data = pd.read_pickle(m5_cache_path)
            if m5_combined is None:
                m5_combined = m5_data
            else:
                m5_combined = pd.concat([m5_combined, m5_data])

        if h1_cache_path.exists():
            h1_data = pd.read_pickle(h1_cache_path)
            if h1_combined is None:
                h1_combined = h1_data
            else:
                h1_combined = pd.concat([h1_combined, h1_data])

    if m5_combined is not None:
        m5_combined = m5_combined.sort_index()

    if h1_combined is not None:
        h1_combined = h1_combined.sort_index()

    return m5_combined, h1_combined

# Function to display analysis results
def display_analysis_results(results):
    st.subheader("Analysis Results")
    st.markdown("### Summary Statistics")

    col1, col2 = st.columns(2)

    with col1:
        up_data = results[results['reference_direction'] == "up"]
        up_success = np.sum(up_data['hit_target_first'])
        up_total = np.sum(up_data['hit_target_first'] | up_data['hit_stoploss_first'])
        up_probability = (up_success / up_total * 100) if up_total > 0 else 0

        st.metric(
            label=f"When first {selected_reference_tf} closes ABOVE {selected_tf} open",
            value=f"{up_probability:.2f}%",
            delta=f"{up_success}/{up_total} cases"
        )

    with col2:
        down_data = results[results['reference_direction'] == "down"]
        down_success = np.sum(down_data['hit_target_first'])
        down_total = np.sum(down_data['hit_target_first'] | down_data['hit_stoploss_first'])
        down_probability = (down_success / down_total * 100) if down_total > 0 else 0

        st.metric(
            label=f"When first {selected_reference_tf} closes BELOW {selected_tf} open",
            value=f"{down_probability:.2f}%",
            delta=f"{down_success}/{down_total} cases"
        )

    st.markdown("### Detailed Analysis by Timeframe Candle")
    direction_filter = st.multiselect(
        "Filter by Direction",
        options=["up", "down"],
        default=["up", "down"]
    )

    filtered_results = results[results['reference_direction'].isin(direction_filter)]
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

    display_cols = ['tf_datetime', 'tf_open', 'first_reference_close', 'reference_direction', 'hit_target_first', 'probability']
    st.dataframe(display_results[display_cols])

    st.markdown("### Visualization")
    results['hour'] = results['tf_datetime'].dt.hour
    hourly_data = []

    for hour in range(24):
        for direction in ['up', 'down']:
            hour_direction_mask = (results['hour'] == hour) & (results['reference_direction'] == direction)
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

    if not hourly_df.empty:
        up_df = hourly_df[hourly_df['direction'] == 'up']
        down_df = hourly_df[hourly_df['direction'] == 'down']

        fig = go.Figure()

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

    if st.button("Cache These Results for Faster Future Access"):
        results_cache_path = CACHE_DIR / f"analysis_results_{selected_tf_code}.pkl"
        results.to_pickle(results_cache_path)
        st.success(f"Results cached to {results_cache_path}")

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
# Main function to run the analysis
def run_analysis(selected_files, selected_tf_code, selected_reference_tf_code, tp_percent, sl_percent, enable_end_of_tf_restriction):
    if not selected_files:
        st.sidebar.error("No files selected. Please select at least one CSV file.")
        return

    # Check if cached data exists for both the selected timeframe and reference timeframe
    all_cached = all(
        (CACHE_DIR / get_cache_filename(filename, selected_tf_code)).exists() and
        (CACHE_DIR / get_cache_filename(filename, selected_reference_tf_code)).exists()
        for filename in selected_files
    )

    if not all_cached:
        st.warning("Some files have not been prepared. Please click 'Prepare & Cache Data' first.")
        return

    with st.spinner("Loading data..."):
        # Load and combine reference timeframe data
        reference_combined = None
        for filename in selected_files:
            reference_cache_path = CACHE_DIR / get_cache_filename(filename, selected_reference_tf_code)
            if reference_cache_path.exists():
                reference_data = pd.read_pickle(reference_cache_path)
                if reference_combined is None:
                    reference_combined = reference_data
                else:
                    reference_combined = pd.concat([reference_combined, reference_data])

        # Load and combine selected timeframe data
        h1_combined = None
        for filename in selected_files:
            h1_cache_path = CACHE_DIR / get_cache_filename(filename, selected_tf_code)
            if h1_cache_path.exists():
                h1_data = pd.read_pickle(h1_cache_path)
                if h1_combined is None:
                    h1_combined = h1_data
                else:
                    h1_combined = pd.concat([h1_combined, h1_data])

    if reference_combined is not None:
        reference_combined = reference_combined.sort_index()

    if h1_combined is not None:
        h1_combined = h1_combined.sort_index()

    results = analyze_candle_batch(h1_combined, reference_combined, tp_percent, sl_percent, enable_end_of_tf_restriction)
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

        display_analysis_results(results)

# Sidebar buttons
prepare_data = st.sidebar.button("Prepare & Cache Data")
start_analysis = st.sidebar.button("Start Analysis")

if prepare_data:
    prepare_and_cache_data(selected_files, selected_tf_code, selected_reference_tf_code)

if start_analysis:
    run_analysis(selected_files, selected_tf_code, selected_reference_tf_code, tp_percent, sl_percent, enable_end_of_tf_restriction)

# Best H1 Candles Analysis
filter_best_candles = st.sidebar.button("Filter by Best H1 Candles")

if filter_best_candles:
    if 'results' not in st.session_state:
        st.warning("Please run the analysis first before filtering for best candles.")
    else:
        results = st.session_state.results
        st.subheader("Best Hourly Candles Analysis (Above 60% Probability)")
        results['hour'] = results['tf_datetime'].dt.hour
        best_hours_data = []

        for hour in range(24):
            for direction in ['up', 'down']:
                mask = (results['hour'] == hour) & (results['reference_direction'] == direction)
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

        best_hours_df = pd.DataFrame(best_hours_data)
        high_prob_hours = best_hours_df[best_hours_df['probability'] > 60].sort_values('probability', ascending=False)

        if len(high_prob_hours) > 0:
            st.markdown("### Hourly Candles with > 60% Probability")
            display_cols = ['hour', 'direction', 'probability', 'success', 'total', 'sample_size']
            formatted_high_prob = high_prob_hours[display_cols].copy()
            formatted_high_prob['probability'] = formatted_high_prob['probability'].round(2).astype(str) + '%'
            formatted_high_prob['success_rate'] = formatted_high_prob['success'].astype(str) + '/' + formatted_high_prob['total'].astype(str)
            st.dataframe(formatted_high_prob[['hour', 'direction', 'probability', 'success_rate', 'sample_size']])

            filtered_indices = []
            for _, row in high_prob_hours.iterrows():
                hour = row['hour']
                direction = row['direction']
                indices = results[(results['hour'] == hour) & (results['reference_direction'] == direction)].index
                filtered_indices.extend(indices)

            high_prob_results = results.loc[filtered_indices]
            st.markdown("### Recalculated Metrics Using Only High Probability Hours")

            col1, col2 = st.columns(2)
            with col1:
                filtered_up_data = high_prob_results[high_prob_results['reference_direction'] == "up"]
                filtered_up_success = np.sum(filtered_up_data['hit_target_first'])
                filtered_up_total = np.sum(filtered_up_data['hit_target_first'] | filtered_up_data['hit_stoploss_first'])
                filtered_up_probability = (filtered_up_success / filtered_up_total * 100) if filtered_up_total > 0 else 0
                st.metric(
                    label=f"When first {selected_reference_tf} closes ABOVE {selected_tf} open (filtered)",
                    value=f"{filtered_up_probability:.2f}%",
                    delta=f"{filtered_up_success}/{filtered_up_total} cases"
                )

            with col2:
                filtered_down_data = high_prob_results[high_prob_results['reference_direction'] == "down"]
                filtered_down_success = np.sum(filtered_down_data['hit_target_first'])
                filtered_down_total = np.sum(filtered_down_data['hit_target_first'] | filtered_down_data['hit_stoploss_first'])
                filtered_down_probability = (filtered_down_success / filtered_down_total * 100) if filtered_down_total > 0 else 0
                st.metric(
                    label=f"When first {selected_reference_tf} closes BELOW {selected_tf} open (filtered)",
                    value=f"{filtered_down_probability:.2f}%",
                    delta=f"{filtered_down_success}/{filtered_down_total} cases"
                )

            st.metric(
                label="Overall Success Rate (High Probability Hours Only)",
                value=f"{((filtered_up_success + filtered_down_success) / (filtered_up_total + filtered_down_total) * 100):.2f}%",
                delta=f"{filtered_up_success + filtered_down_success}/{filtered_up_total + filtered_down_total} cases"
            )

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
            filtered_csv_data = high_prob_results.to_csv(index=False)
            st.download_button(
                label="Download High Probability Results as CSV",
                data=filtered_csv_data,
                file_name=f"high_probability_{selected_tf_code}_results.csv",
                mime="text/csv"
            )
        else:
            st.warning("No hourly candles found with probability above 60%. Try lowering the threshold or analyzing more data.")