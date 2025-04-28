import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib
import altair as alt

# Set the page configuration and title
st.set_page_config(page_title="Sales Predictor", layout="wide")
st.title("\U0001F4CA Smart Sales Prediction Dashboard")

# --- Load the trained model and preprocessed data ---
model = joblib.load("best_sales_model.joblib")  # Load the pre-trained XGBoost model
df = pd.read_csv("full_dataset.csv")  # Load the preprocessed dataset
shap_values = joblib.load('shap_values.joblib')  # Load the precomputed SHAP values

# --- Sidebar Inputs ---
st.sidebar.header("Customize Prediction")

# Day of the week selection
day_index = st.sidebar.selectbox(
    "\U0001F4C5 Day of Week",
    options=[(0, "Monday"), (1, "Tuesday"), (2, "Wednesday"),
             (3, "Thursday"), (4, "Friday"), (5, "Saturday"), (6, "Sunday")],
    format_func=lambda x: x[1]
)

# Is it a holiday?
is_holiday = st.sidebar.selectbox("\U0001F3D6️ Is it a Holiday?", [0, 1], format_func=lambda x: "Yes" if x else "No")

# Weekend or weekday check
is_weekend = 1 if day_index[0] in [5, 6] else 0

# --- Traffic Range Input ---
traffic_min = int(df['Value_traffic'].min())
traffic_max = int(df['Value_traffic'].max())
traffic_step = 100

traffic_ranges = [(f"{i}-{i+traffic_step}", i, i+traffic_step) for i in range(traffic_min, traffic_max, traffic_step)]
traffic_range = st.sidebar.selectbox("🚗 Select Traffic Range", options=[x[0] for x in traffic_ranges])
lower_traffic, upper_traffic = next((low, high) for label, low, high in traffic_ranges if f"{low}-{high}" == traffic_range)
avg_traffic = (lower_traffic + upper_traffic) / 2

# --- Prediction for Total Sales ---
total_daily_sales = 0
for hour in range(24):
    input_data = pd.DataFrame([{
        "is_holiday": is_holiday,
        "is_weekend": is_weekend,
        "day_of_week": day_index[0],
        "Value_traffic": avg_traffic,
        "hour": hour
    }])
    prediction = model.predict(input_data)[0]
    total_daily_sales += prediction

st.subheader(f"\U0001F9EE Predicted Total Daily Sales: {total_daily_sales:.2f}")

# --- Filtered Data with Predictions ---
filtered_df = df[
    (df['day_of_week'] == day_index[0]) &
    (df['is_holiday'] == is_holiday) &
    (df['Value_traffic'] >= lower_traffic) &
    (df['Value_traffic'] <= upper_traffic)
].copy()

filtered_df['predicted_sales'] = model.predict(filtered_df[['is_holiday', 'is_weekend', 'day_of_week', 'Value_traffic', 'hour']])
filtered_df['predicted_sales'] = filtered_df['predicted_sales'].round(2)

# --- Graphs Section ---
st.markdown("### \U0001F4CA Visual Insights")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### ⏱️ Predicted Sales by Time Block (Total Daily Count)")

    # Reuse hourly predictions for consistency
    hourly_preds = []
    for hour in range(24):
        # Check if there's any data for this hour in the dataset
        matching_rows = df[df['hour'] == hour]
        if not matching_rows.empty:
            # Get the time block for the first row of the matching data
            time_block = matching_rows['time_block'].iloc[0]
        else:
            # Handle the case where no data exists for this hour (e.g., set a default value)
            time_block = 'No sales'
        
        input_data = pd.DataFrame([{
            "is_holiday": is_holiday,
            "is_weekend": is_weekend,
            "day_of_week": day_index[0],
            "Value_traffic": avg_traffic,
            "hour": hour
        }])
        prediction = model.predict(input_data)[0]
        hourly_preds.append({'hour': hour, 'time_block': time_block, 'predicted_sales': prediction})

    block_df = pd.DataFrame(hourly_preds)
    block_sales = block_df.groupby('time_block')['predicted_sales'].sum().reset_index()
    block_sales['predicted_sales'] = block_sales['predicted_sales'].round(2)

    block_chart = alt.Chart(block_sales).mark_bar().encode(
        x=alt.X('time_block:N', sort=['Morning', 'Afternoon', 'Evening', 'Night']),
        y=alt.Y('predicted_sales:Q', title='Total Predicted Sales'),
        color='time_block:N',
        tooltip=['time_block:N', 'predicted_sales:Q']
    ).properties(width=400, height=350)

    st.altair_chart(block_chart)

    st.markdown(""" ⏰ Time Block Definitions""")
    st.markdown("""- **Morning**: 06:00 AM - 11:59 AM  
                - **Afternoon**: 12:00 PM - 05:59 PM  
                - **Evening**: 06:00 PM - 09:59 PM  
                - **Night**: 10:00 PM - 05:59 AM""")

with col2:
    st.markdown("#### 🕒 Predicted Sales by Hour (Total Daily Breakdown)")

    # Reuse hourly predictions for consistency
    hour_df = pd.DataFrame(hourly_preds)  # This uses the same 24 predictions from the loop above
    hour_df['predicted_sales'] = hour_df['predicted_sales'].round(2)

    # Line + labels chart
    line = alt.Chart(hour_df).mark_line(point=True).encode(
        x=alt.X('hour:O', title='Hour of Day'),
        y=alt.Y('predicted_sales:Q', title='Predicted Sales'),
        tooltip=[alt.Tooltip('hour:O', title='Hour'), alt.Tooltip('predicted_sales:Q', title='Predicted Sales')]
    )

    labels = alt.Chart(hour_df).mark_text(
        align='center',
        baseline='bottom',
        dy=-10,
        fontSize=11,
        color='white'
    ).encode(
        x='hour:O',
        y='predicted_sales:Q',
        text=alt.Text('predicted_sales:Q', format=".2f")  # format label to 2 decimal places
    )

    st.altair_chart((line + labels).properties(width=400, height=350))

st.markdown("#### 🚦 Optimal Traffic Level for Maximum Hourly Sales")
with st.expander("Analysed Graph is here 👇🏻"):
    traffic_range_values = list(range(max(50, traffic_min), traffic_max + 1, 100))  # Start from 1 to avoid 0 traffic
    optimal_traffic = []

    for hour in range(24):
        best_traffic = None
        max_sales = -np.inf

        for traffic in traffic_range_values:
            test_input = pd.DataFrame([{
                "is_holiday": is_holiday,
                "is_weekend": is_weekend,
                "day_of_week": day_index[0],
                "Value_traffic": traffic,
                "hour": hour
            }])
            prediction = model.predict(test_input)[0]

            if prediction > max_sales:
                max_sales = prediction
                best_traffic = traffic

        optimal_traffic.append({
            "hour": hour,
            "optimal_traffic": best_traffic,
            "max_predicted_sales": round(max_sales, 2)
        })

    optimal_df = pd.DataFrame(optimal_traffic)

    # Chart for Max Predicted Sales by Hour
    opt_chart = alt.Chart(optimal_df).mark_line(point=True).encode(
        x=alt.X('hour:O', title='Hour of Day'),
        y=alt.Y('max_predicted_sales:Q', title='Max Predicted Sales'),
        color=alt.value("#4C78A8"),  # Consistent color
        tooltip=[alt.Tooltip('hour:O', title='Hour'), alt.Tooltip('max_predicted_sales:Q', title='Max Predicted Sales')]
    ).properties(
        width=650,
        height=350,
        title="🚦 Maximum Predicted Sales and Optimal Traffic by Hour"
    )

    # Annotate each point with optimal traffic
    text_labels = alt.Chart(optimal_df).mark_text(
        align='center',
        baseline='bottom',
        dy=-7,
        fontSize=11,
        fontWeight="bold",
        color='white'
    ).encode(
        x='hour:O',
        y='max_predicted_sales:Q',
        text=alt.Text('optimal_traffic:Q', format='.0f')
    )

    st.altair_chart(opt_chart + text_labels)

# --- SHAP Explainability ---
st.markdown("### 🔍 Feature Impact (Peak Hour)")

hourly_data = pd.DataFrame([{
    "is_holiday": is_holiday,
    "is_weekend": is_weekend,
    "day_of_week": day_index[0],
    "Value_traffic": avg_traffic,
    "hour": hour
} for hour in range(24)])

hourly_data["predicted_sales"] = model.predict(hourly_data)

# Identify the hour with the maximum predicted sales
peak_row = hourly_data.loc[hourly_data["predicted_sales"].idxmax()]
peak_hour = int(peak_row["hour"])

# Prepare sample input for SHAP
sample_input = pd.DataFrame([{
    "is_holiday": is_holiday,
    "is_weekend": is_weekend,
    "day_of_week": day_index[0],
    "Value_traffic": avg_traffic,
    "hour": peak_hour
}])

# Run SHAP Explainer
explainer = shap.Explainer(model)
shap_val = explainer(sample_input)

# Plot SHAP Waterfall
fig = plt.figure(figsize=(6, 4))
shap.plots.waterfall(shap_val[0], show=False)
st.pyplot(fig, bbox_inches='tight', dpi=300, pad_inches=0.2)

# Annotation
st.markdown(f"🕒 **Peak Hour**: {peak_hour}:00  |  🚗 **Traffic Input**: {int(avg_traffic)}  |  🔢 **Max Predicted Sales**: {peak_row['predicted_sales']:.2f}")
st.caption("🧠 The SHAP chart explains how each feature influenced the predicted sales during the busiest hour.")