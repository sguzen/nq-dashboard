import pandas as pd
import streamlit as st

import pandas as pd

def analyze_candle_batch(h1_data, reference_data, selected_reference_tf_code, tp_percent, sl_percent, enable_end_of_tf_restriction, enable_dynamic_sl):
    batch_results = []

    for i, (tf_time, tf_row) in enumerate(h1_data.iterrows()):
        tf_end_time = tf_time + pd.Timedelta(hours=1) - pd.Timedelta(seconds=1)
        reference_candles_in_tf = reference_data.loc[tf_time:tf_end_time]

        if len(reference_candles_in_tf) > 0:
            first_reference_candle = reference_candles_in_tf.iloc[0]
            candle_size = first_reference_candle['high'] - first_reference_candle['low']
            candle_size_percentage = (candle_size / first_reference_candle['open']) * 100

            if (selected_reference_tf_code == '5T' and candle_size_percentage < 0.15) or selected_reference_tf_code != '5T':       
                reference_direction = "up" if first_reference_candle['close'] > first_reference_candle['open'] else "down"

                # REVERSED TARGET/STOP LOGIC FOR BULLISH CANDLES
                if reference_direction == "up":
                    target_level = first_reference_candle['close'] * (1 - tp_percent / 100)  # TP BELOW close
                    if enable_dynamic_sl and i > 0:
                        previous_candle = h1_data.iloc[i - 1]
                        previous_candle_range = previous_candle['high'] - previous_candle['low']
                        dynamic_sl_level = first_reference_candle['close'] + (0.5 * previous_candle_range)
                        user_sl_level = first_reference_candle['close'] * (1 + sl_percent / 100)  # SL ABOVE close
                        stop_level = min(dynamic_sl_level, user_sl_level)  # Tighter stop
                    else:
                        stop_level = first_reference_candle['close'] * (1 + sl_percent / 100)
                else:  # Bearish (unchanged)
                    target_level = first_reference_candle['close'] * (1 + tp_percent / 100)
                    if enable_dynamic_sl and i > 0:
                        previous_candle = h1_data.iloc[i - 1]
                        previous_candle_range = previous_candle['high'] - previous_candle['low']
                        dynamic_sl_level = first_reference_candle['close'] - (0.5 * previous_candle_range)
                        user_sl_level = first_reference_candle['close'] * (1 - sl_percent / 100)
                        stop_level = max(dynamic_sl_level, user_sl_level)
                    else:
                        stop_level = first_reference_candle['close'] * (1 - sl_percent / 100)

                next_candles = reference_candles_in_tf.iloc[1:]
                hit_target = False
                hit_stop = False
                time_to_hit = None

                # REVERSED MAE/MFE CALCULATIONS FOR BULLISH CANDLES
                entry_price = first_reference_candle['close']
                if reference_direction == "up":
                    mae = ((entry_price - tf_row['low']) / entry_price * 100)  # Worst downside move
                    mfe = ((entry_price - tf_row['high']) / entry_price * 100)  # Best upside move
                else:
                    mae = ((tf_row['high'] - entry_price) / entry_price * 100)  # Worst upside move
                    mfe = ((tf_row['low'] - entry_price) / entry_price * 100)   # Best downside move

                # REVERSED HIT DETECTION FOR BULLISH CANDLES
                for _, candle in next_candles.iterrows():
                    if reference_direction == "up":
                        if candle['low'] <= target_level:  # Price falls to TP
                            hit_target = True
                            time_to_hit = (candle.name - tf_time).total_seconds() / 60
                            break
                        if candle['high'] >= stop_level:   # Price rises to SL
                            hit_stop = True
                            time_to_hit = (candle.name - tf_time).total_seconds() / 60
                            break
                    else:  # Bearish (unchanged)
                        if candle['low'] <= target_level:
                            hit_target = True
                            time_to_hit = (candle.name - tf_time).total_seconds() / 60
                            break
                        if candle['high'] >= stop_level:
                            hit_stop = True
                            time_to_hit = (candle.name - tf_time).total_seconds() / 60
                            break

                if enable_end_of_tf_restriction and not hit_target and not hit_stop:
                    hit_stop = True
                    time_to_hit = (tf_end_time - tf_time).total_seconds() / 60

                batch_results.append({
                    'tf_datetime': tf_time,
                    'tf_open': tf_row['open'],
                    'first_reference_close': entry_price,
                    'reference_direction': reference_direction,
                    'hit_target_first': hit_target,
                    'hit_stoploss_first': hit_stop,
                    'day_of_week': tf_time.strftime('%A'),
                    'mae': mae,
                    'mfe': mfe,
                    'time_to_hit': time_to_hit,
                    'probability': None
                })

    return pd.DataFrame(batch_results)