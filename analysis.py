import pandas as pd
import streamlit as st
import pandas as pd

def calculate_trade_levels(first_reference_candle, tp_percent, sl_percent, enable_reverse_calculation):
    """Calculate target and stop levels based on candle direction and reverse settings."""
    is_bullish = first_reference_candle['close'] > first_reference_candle['open']
    
    if is_bullish:
        if enable_reverse_calculation:
            return {
                'target_level': first_reference_candle['close'] * (1 - tp_percent / 100),
                'stop_level': first_reference_candle['close'] * (1 + sl_percent / 100),
                'reference_direction': "up",
                'trade_direction': "down"
            }
        else:
            return {
                'target_level': first_reference_candle['close'] * (1 + tp_percent / 100),
                'stop_level': first_reference_candle['close'] * (1 - sl_percent / 100),
                'reference_direction': "up",
                'trade_direction': "down"
            }
    else:
        if enable_reverse_calculation:
            return {
                'target_level': first_reference_candle['close'] * (1 + tp_percent / 100),
                'stop_level': first_reference_candle['close'] * (1 - sl_percent / 100),
                'reference_direction': "down",
                'trade_direction': "up"
            }
        else:
            return {
                'target_level': first_reference_candle['close'] * (1 - tp_percent / 100),
                'stop_level': first_reference_candle['close'] * (1 + sl_percent / 100),
                'reference_direction': "down",
                'trade_direction': "up"
            }
        
def analyze_candle_batch(h1_data, reference_data, selected_reference_tf_code, tp_percent, sl_percent, enable_end_of_tf_restriction, enable_reverse_calculation):
    batch_results = []

    for i, (tf_time, tf_row) in enumerate(h1_data.iterrows()):
        tf_end_time = tf_time + pd.Timedelta(hours=1) - pd.Timedelta(seconds=1)
        reference_candles_in_tf = reference_data.loc[tf_time:tf_end_time]

        if len(reference_candles_in_tf) > 0:
            first_reference_candle = reference_candles_in_tf.iloc[0]
            candle_size = first_reference_candle['high'] - first_reference_candle['low']
            candle_size_percentage = (candle_size / first_reference_candle['open']) * 100

            if (selected_reference_tf_code == '5T' and candle_size_percentage < 0.15) or selected_reference_tf_code != '5T':
                result = calculate_trade_levels(
                    first_reference_candle,
                    tp_percent,
                    sl_percent,
                    enable_reverse_calculation
                )
                target_level = result['target_level']
                stop_level = result['stop_level']
                trade_direction = result['trade_direction']
                reference_direction = result['reference_direction']

                next_candles = reference_candles_in_tf.iloc[1:]
                hit_target = False
                hit_stop = False
                time_to_hit = None

                entry_price = first_reference_candle['close']
                if reference_direction == "up" and enable_reverse_calculation:
                    mae = ((entry_price - tf_row['low']) / entry_price * 100)  
                    mfe = ((entry_price - tf_row['high']) / entry_price * 100)  
                if reference_direction == "down" and enable_reverse_calculation:
                    mae = ((tf_row['high'] - entry_price) / entry_price * 100)  
                    mfe = ((tf_row['low'] - entry_price) / entry_price * 100)   
                if reference_direction == "up" and not enable_reverse_calculation:
                    mae = ((tf_row['low'] - entry_price) / entry_price * 100)
                    mfe = ((tf_row['high'] - entry_price) / entry_price * 100) 
                if reference_direction == "down" and not enable_reverse_calculation:
                    mae = ((entry_price - tf_row['high']) / entry_price * 100)
                    mfe = ((entry_price - tf_row['low']) / entry_price * 100)

                for _, candle in next_candles.iterrows():
                    if reference_direction == "up" and enable_reverse_calculation:
                        if candle['low'] <= target_level:  # Price falls to TP
                            hit_target = True
                            time_to_hit = (candle.name - tf_time).total_seconds() / 60
                            break
                        if candle['high'] >= stop_level:   # Price rises to SL
                            hit_stop = True
                            time_to_hit = (candle.name - tf_time).total_seconds() / 60
                            break
                    if reference_direction == "down" and enable_reverse_calculation:
                        if candle['high'] >= target_level:
                            hit_target = True
                            time_to_hit = (candle.name - tf_time).total_seconds() / 60
                            break
                        if candle['low'] <= stop_level:
                            hit_stop = True
                            time_to_hit = (candle.name - tf_time).total_seconds() / 60
                            break
                    if reference_direction == "up" and not enable_reverse_calculation:
                        if candle['high'] >= target_level:
                            hit_target = True
                            time_to_hit = (candle.name - tf_time).total_seconds() / 60  # Time in minutes
                            break
                        if candle['low'] <= stop_level:
                            hit_stop = True
                            time_to_hit = (candle.name - tf_time).total_seconds() / 60  # Time in minutes
                            break
                    if reference_direction == "down" and  not enable_reverse_calculation:
                        if candle['low'] <= target_level:
                            hit_target = True
                            time_to_hit = (candle.name - tf_time).total_seconds() / 60  # Time in minutes
                            break
                        if candle['high'] >= stop_level:
                            hit_stop = True
                            time_to_hit = (candle.name - tf_time).total_seconds() / 60  # Time in minutes
                            break

                if enable_end_of_tf_restriction and not hit_target and not hit_stop:
                    hit_stop = True
                    time_to_hit = (tf_end_time - tf_time).total_seconds() / 60

                batch_results.append({
                    'tf_datetime': tf_time,
                    'tf_open': tf_row['open'],
                    'first_reference_close': entry_price,
                    'reference_direction': reference_direction,
                    'trade_direction': trade_direction,
                    'hit_target_first': hit_target,
                    'hit_stoploss_first': hit_stop,
                    'day_of_week': tf_time.strftime('%A'),
                    'hour_of_day': tf_time.strftime("%I"),
                    'mae': mae,
                    'mfe': mfe,
                    'time_to_hit': time_to_hit,
                    'probability': None
                })

    return pd.DataFrame(batch_results)
