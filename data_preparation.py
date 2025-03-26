import streamlit as st
import pandas as pd
import utils
import numpy as np
import plotly.graph_objects as go
import display # refactor this, we don't need this library here

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
            file_path = utils.DATA_DIR / filename
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
            df_processed = utils.convert_to_ny_local_time(df_processed, timestamp_column='datetime')

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

            m1_cache_path = utils.CACHE_DIR / utils.get_cache_filename(filename, "1T")
            m1_data.to_pickle(m1_cache_path)

            # Cache reference timeframe data (optional, if needed for other purposes)
            reference_data = df_processed.resample(selected_reference_tf_code).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            reference_cache_path = utils.CACHE_DIR / utils.get_cache_filename(filename, selected_reference_tf_code)
            reference_data.to_pickle(reference_cache_path)

            # Cache selected timeframe data
            selected_tf_data = df_processed.resample(selected_tf_code).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            selected_tf_cache_path = utils.CACHE_DIR / utils.get_cache_filename(filename, selected_tf_code)
            selected_tf_data.to_pickle(selected_tf_cache_path)

        except Exception as e:
            st.sidebar.error(f"Error processing {filename}: {str(e)}")

        prep_progress.progress((i + 1) / len(selected_files))

    prep_status.text("Data preparation complete!")
    st.sidebar.success(f"Processed and cached data for {len(selected_files)} files")

# Function to load and combine cached data
def load_and_combine_cached_data(selected_files, selected_tf_code):
    m5_combined = None
    h1_combined = None

    for filename in selected_files:
        m5_cache_path = utils.CACHE_DIR / utils.get_cache_filename(filename, "5T")
        h1_cache_path = utils.CACHE_DIR / utils.get_cache_filename(filename, selected_tf_code)

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

def check_results_available():
    if 'results' not in st.session_state:
        st.warning("Please run the analysis first before filtering for best candles.")
        return False
    return True

def calculate_best_hours_data(results):
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

    return pd.DataFrame(best_hours_data)

def filter_high_probability_hours(best_hours_df):
    if 'probability' in best_hours_df.columns:
        return best_hours_df[best_hours_df['probability'] > 60].sort_values('probability', ascending=False)
    else:
        st.error("Probability column not found in the results. Please check the data preparation step.")
        return pd.DataFrame()

def get_filtered_indices(results, high_prob_hours):
    filtered_indices = []
    for _, row in high_prob_hours.iterrows():
        hour = row['hour']
        direction = row['direction']
        indices = results[(results['hour'] == hour) & (results['reference_direction'] == direction)].index
        filtered_indices.extend(indices)
    return results.loc[filtered_indices]

def filter_best_candles(selected_reference_tf, selected_tf, selected_tf_code):
    if not check_results_available():
        return

    results = st.session_state.results
    st.subheader("Best Hourly Candles Analysis (Above 60% Probability)")

    best_hours_df = calculate_best_hours_data(results)
    high_prob_hours = filter_high_probability_hours(best_hours_df)
    display.display_high_probability_hours(high_prob_hours)

    if len(high_prob_hours) > 0:
        high_prob_results = get_filtered_indices(results, high_prob_hours)
        display.display_recalculated_metrics(high_prob_results, selected_reference_tf, selected_tf)
        display.visualize_high_probability_hours(high_prob_hours)
        utils.download_high_probability_results(high_prob_results, selected_tf_code)    