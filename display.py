import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import utils

# Function to display analysis results
def display_analysis_results(results, selected_tf_code):
    st.subheader("Analysis Results")
    st.markdown("### Summary Statistics")

    # Overall success rate
    overall_success = np.sum(results['hit_target_first'])
    overall_total = np.sum(results['hit_target_first'] | results['hit_stoploss_first'])
    overall_probability = (overall_success / overall_total * 100) if overall_total > 0 else 0

    st.metric(
        label="Overall Success Rate",
        value=f"{overall_probability:.2f}%",
        delta=f"{overall_success}/{overall_total} cases"
    )

    # Overall MAE and MFE
    overall_mae = results['mae'].mean()
    overall_mfe = results['mfe'].mean()

    st.metric(
        label="Overall Average MAE",
        value=f"{overall_mae:.2f}%"
    )
    st.metric(
        label="Overall Average MFE",
        value=f"{overall_mfe:.2f}%"
    )

    # Overall average time to hit target or stop
    overall_avg_time = results['time_to_hit'].mean()

    st.metric(
        label="Overall Average Time to Hit Target or Stop",
        value=f"{overall_avg_time:.2f} minutes"
    )

    # Day-of-week breakdown
    st.markdown("### Success Rate, MAE, and MFE by Day of the Week")
    day_of_week_data = []

    for day in results['day_of_week'].unique():
        day_mask = (results['day_of_week'] == day)
        day_data = results[day_mask]
        if len(day_data) > 0:
            day_success = np.sum(day_data['hit_target_first'])
            day_total = np.sum(day_data['hit_target_first'] | day_data['hit_stoploss_first'])
            day_probability = (day_success / day_total * 100) if day_total > 0 else 0
            day_mae = day_data['mae'].mean()
            day_mfe = day_data['mfe'].mean()
            day_of_week_data.append({
                'day_of_week': day,
                'success': day_success,
                'total': day_total,
                'probability': day_probability,
                'mae': day_mae,
                'mfe': day_mfe
            })

    day_of_week_df = pd.DataFrame(day_of_week_data)

    if not day_of_week_df.empty:
        # Display table
        st.dataframe(day_of_week_df[['day_of_week', 'probability', 'mae', 'mfe', 'success', 'total']])

        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=day_of_week_df['day_of_week'],
            y=day_of_week_df['probability'],
            name='Probability',
            marker_color='blue',
            opacity=0.7,
            text=day_of_week_df['success'].astype(str) + '/' + day_of_week_df['total'].astype(str),
            hoverinfo='text'
        ))
        fig.add_trace(go.Bar(
            x=day_of_week_df['day_of_week'],
            y=day_of_week_df['mae'],
            name='MAE',
            marker_color='red',
            opacity=0.7,
            hoverinfo='y'
        ))
        fig.add_trace(go.Bar(
            x=day_of_week_df['day_of_week'],
            y=day_of_week_df['mfe'],
            name='MFE',
            marker_color='green',
            opacity=0.7,
            hoverinfo='y'
        ))

        fig.update_layout(
            title='Performance by Day of the Week',
            xaxis_title='Day of the Week',
            yaxis_title='Value (%)',
            barmode='group',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough data for day-of-week breakdown.")

    # Rest of the analysis (hourly breakdown, etc.)
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

    display_cols = ['tf_datetime', 'tf_open', 'first_reference_close', 'reference_direction', 'hit_target_first', 'mae', 'mfe', 'probability']
    st.dataframe(display_results[display_cols])

    # Hourly visualization (existing code)
    st.markdown("### Visualization by Hour of Day")
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

    # Export and caching options (existing code)
    if st.button("Cache These Results for Faster Future Access"):
        results_cache_path = utils.CACHE_DIR / f"analysis_results_{selected_tf_code}.pkl"
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

def display_high_probability_hours(high_prob_hours):
    if len(high_prob_hours) > 0:
        st.markdown("### Hourly Candles with > 60% Probability")
        display_cols = ['hour', 'direction', 'probability', 'success', 'total', 'sample_size']
        formatted_high_prob = high_prob_hours[display_cols].copy()
        formatted_high_prob['probability'] = formatted_high_prob['probability'].round(2).astype(str) + '%'
        formatted_high_prob['success_rate'] = formatted_high_prob['success'].astype(str) + '/' + formatted_high_prob['total'].astype(str)
        st.dataframe(formatted_high_prob[['hour', 'direction', 'probability', 'success_rate', 'sample_size']])
    else:
        st.warning("No hourly candles found with probability above 60%. Try lowering the threshold or analyzing more data.")

def display_recalculated_metrics(high_prob_results, selected_reference_tf, selected_tf):
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

def visualize_high_probability_hours(high_prob_hours):
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
