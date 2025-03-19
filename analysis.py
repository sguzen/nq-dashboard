import pandas as pd
import streamlit as st

# Function to analyze candle batch
def analyze_candle_batch(h1_data, reference_data, selected_reference_tf_code, tp_percent, sl_percent, enable_end_of_tf_restriction, enable_dynamic_sl):
    batch_results = []

    for i, (tf_time, tf_row) in enumerate(h1_data.iterrows()):
        tf_end_time = tf_time + pd.Timedelta(hours=1) - pd.Timedelta(seconds=1)
        reference_candles_in_tf = reference_data.loc[tf_time:tf_end_time]

        if len(reference_candles_in_tf) > 0:
            first_reference_candle = reference_candles_in_tf.iloc[0]
            # Check if the first reference candle's size is less than 0.16%
            # If m5 is not selected, this calculation will be done for no purpose, refactor it
            candle_size = first_reference_candle['high'] - first_reference_candle['low']
            candle_size_percentage = (candle_size / first_reference_candle['open']) * 100

            if (selected_reference_tf_code == '5T' and candle_size_percentage < 0.15) or selected_reference_tf_code != '5T':       
                reference_direction = "up" if first_reference_candle['close'] > first_reference_candle['open'] else "down"

                # Calculate target level
                if reference_direction == "up":
                    target_level = first_reference_candle['close'] * (1 + tp_percent / 100)
                else:
                    target_level = first_reference_candle['close'] * (1 - tp_percent / 100)

                # Calculate stop loss level
                if enable_dynamic_sl and i > 0:  # Ensure there is a previous candle
                    previous_candle = h1_data.iloc[i - 1]
                    previous_candle_range = previous_candle['high'] - previous_candle['low']
                    dynamic_sl_level = first_reference_candle['close'] - (0.5 * previous_candle_range) if reference_direction == "up" else first_reference_candle['close'] + (0.5 * previous_candle_range)
                    user_sl_level = first_reference_candle['close'] * (1 - sl_percent / 100) if reference_direction == "up" else first_reference_candle['close'] * (1 + sl_percent / 100)

                    # Use the closer of the two SL levels
                    if reference_direction == "up":
                        stop_level = max(dynamic_sl_level, user_sl_level)
                    else:
                        stop_level = min(dynamic_sl_level, user_sl_level)
                else:
                    # Use the user-defined SL level
                    if reference_direction == "up":
                        stop_level = first_reference_candle['close'] * (1 - sl_percent / 100)
                    else:
                        stop_level = first_reference_candle['close'] * (1 + sl_percent / 100)

                next_candles = reference_candles_in_tf.iloc[1:]
                hit_target = False
                hit_stop = False
                time_to_hit = None  # Initialize time_to_hit

                # Initialize MAE and MFE
                entry_price = first_reference_candle['close']
                mae = 0
                mfe = 0
                
                # Calculate MAE and MFE based on the current H1 candle (tf_row)
                if reference_direction == "up":
                    # For "up" direction, MAE is the lowest point compared to entry
                    mae = ((tf_row['low'] - entry_price) / entry_price * 100)
                    # For "up" direction, MFE is the highest point compared to entry
                    mfe = ((tf_row['high'] - entry_price) / entry_price * 100)
                else:
                    # For "down" direction, MAE is the highest point compared to entry
                    mae = ((entry_price - tf_row['high']) / entry_price * 100)
                    # For "down" direction, MFE is the lowest point compared to entry
                    mfe = ((entry_price - tf_row['low']) / entry_price * 100)

                for _, candle in next_candles.iterrows():
                    if reference_direction == "up":
                        # Check if target or stop level is hit
                        if candle['high'] >= target_level:
                            hit_target = True
                            time_to_hit = (candle.name - tf_time).total_seconds() / 60  # Time in minutes
                            break
                        if candle['low'] <= stop_level:
                            hit_stop = True
                            time_to_hit = (candle.name - tf_time).total_seconds() / 60  # Time in minutes
                            break
                    else:
                        # Check if target or stop level is hit
                        if candle['low'] <= target_level:
                            hit_target = True
                            time_to_hit = (candle.name - tf_time).total_seconds() / 60  # Time in minutes
                            break
                        if candle['high'] >= stop_level:
                            hit_stop = True
                            time_to_hit = (candle.name - tf_time).total_seconds() / 60  # Time in minutes
                            break

                # Handle end-of-timeframe restriction based on the toggle switch
                if enable_end_of_tf_restriction and not hit_target and not hit_stop:
                    hit_stop = True  # Mark as a loser if neither target nor stop level is hit by the end of the larger timeframe
                    time_to_hit = (tf_end_time - tf_time).total_seconds() / 60  # Time in minutes

                hit_target_first = hit_target
                hit_stoploss_first = hit_stop

                batch_results.append({
                    'tf_datetime': tf_time,
                    'tf_open': tf_row['open'],
                    'first_reference_close': first_reference_candle['close'],
                    'reference_direction': reference_direction,
                    'hit_target_first': hit_target_first,
                    'hit_stoploss_first': hit_stoploss_first,
                    'day_of_week': tf_time.strftime('%A'),  # Add day of the week
                    'mae': mae,
                    'mfe': mfe,
                    'time_to_hit': time_to_hit,  # Add time_to_hit column
                    'probability': None
                })

    return pd.DataFrame(batch_results)
