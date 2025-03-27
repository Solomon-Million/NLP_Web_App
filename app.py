import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import qrcode
from io import BytesIO
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data_path = "Store/merged_data_final.csv"  # Update if needed
df = pd.read_csv(data_path)

# Create a dropdown format: "Shortname (Longname) - tic"
df['company_display'] = df['Shortname'] + " (" + df['Longname'] + ") - " + df['tic']


# 🎨 App Layout
st.title("📊 Stock & Sentiment Analysis Web App")
st.markdown("""
Welcome to the **Stock & Sentiment Analysis Web App**!  

This tool analyzes **sentiment trends in ITEM 1A (Risk Factors) sections of 10-K financial reports**  
and compares them with **historical stock price changes** for S&P 500 companies.

### 🔍 How to Use:
- **🔹 Select a company** to explore its stock & sentiment trends.
- **📅 Choose a time range** to filter the data across all views.
- **🏢 Compare by sector or industry** to see broad trends.
- **📈 Check correlation analysis** to understand sentiment & stock movement relationships.
""")


# 📌 **Step 1: Select a Company (Combined Search & Dropdown)**
st.sidebar.header("🔍 Search or Select a Company")

# Allow both search & dropdown selection
selected_company = st.sidebar.selectbox(
    "Start typing or select from the list:",
    options=df["company_display"].unique()
)

selected_tic = selected_company.split(" - ")[-1]

# 📌 **Step 2: Select Time Range**
st.sidebar.header("📅 Select Time Range")
year_range = st.sidebar.slider("Choose Year Range:", 2007, 2023, (2007, 2023))

# Filter the dataset by the selected time range
df = df[(df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1])]

# 📊 **Company-Level Analysis**
st.subheader(f"📊 {selected_company} - Stock vs. Sentiment Trends")
company_data = df[df['tic'] == selected_tic].copy()

fig = go.Figure()

# Stock % Change
fig.add_trace(go.Scatter(
    x=company_data["Year"], y=company_data["Stock_Change%"],
    mode="lines+markers", name="Stock % Change", line=dict(color="blue", width=2), yaxis="y1"
))

# Sentiment Score
fig.add_trace(go.Scatter(
    x=company_data["Year"], y=company_data["Sentiment_Score"],
    mode="lines+markers", name="Sentiment Score", line=dict(color="red", width=2, dash="dash"), yaxis="y2"
))

# Layout
fig.update_layout(
    xaxis={"title": "Year"},
    yaxis={"title": "Stock Price Change (%)"},
    yaxis2={"title": "Sentiment Score", "overlaying": "y", "side": "right"},
    hovermode="x"
)
st.plotly_chart(fig)

# 📊 **Sector-Level Analysis (Now Starts Empty)**
st.subheader("📊 Sector-Wide Sentiment vs. Stock Performance")

# Let users manually add sectors instead of showing all by default
selected_sectors = st.multiselect("Select Sectors to Display:", df["Sector"].unique())

if selected_sectors:
    sector_data = df[df["Sector"].isin(selected_sectors)].copy()
    sector_trends = sector_data.groupby(["Year", "Sector"]).agg({"Stock_Change%": "mean", "Sentiment_Score": "mean"}).reset_index()

    fig_sector = go.Figure()

    for sector in selected_sectors:
        sec_df = sector_trends[sector_trends["Sector"] == sector]
        fig_sector.add_trace(go.Scatter(x=sec_df["Year"], y=sec_df["Stock_Change%"], mode="lines+markers", name=f"{sector} - Stock % Change"))
        fig_sector.add_trace(go.Scatter(x=sec_df["Year"], y=sec_df["Sentiment_Score"], mode="lines+markers", name=f"{sector} - Sentiment Score", yaxis="y2"))

    fig_sector.update_layout(
        xaxis={"title": "Year"},
        yaxis={"title": "Avg. Stock Price Change (%)"},
        yaxis2={"title": "Avg. Sentiment Score", "overlaying": "y", "side": "right"},
        hovermode="x"
    )
    st.plotly_chart(fig_sector)
else:
    st.warning("🔍 Select at least one sector to display the analysis.")

# 📊 **Industry-Level Analysis (Now Starts Empty)**
st.subheader("🏭 Industry-Wide Sentiment vs. Stock Performance")

# Let users manually add industries instead of showing all by default
selected_industries = st.multiselect("Select Industries to Display:", df["Industry"].unique())

if selected_industries:
    industry_data = df[df["Industry"].isin(selected_industries)].copy()
    industry_trends = industry_data.groupby(["Year", "Industry"]).agg({"Stock_Change%": "mean", "Sentiment_Score": "mean"}).reset_index()

    fig_industry = go.Figure()

    for industry in selected_industries:
        ind_df = industry_trends[industry_trends["Industry"] == industry]
        fig_industry.add_trace(go.Scatter(x=ind_df["Year"], y=ind_df["Stock_Change%"], mode="lines+markers", name=f"{industry} - Stock % Change"))
        fig_industry.add_trace(go.Scatter(x=ind_df["Year"], y=ind_df["Sentiment_Score"], mode="lines+markers", name=f"{industry} - Sentiment Score", yaxis="y2"))

    fig_industry.update_layout(
        xaxis={"title": "Year"},
        yaxis={"title": "Avg. Stock Price Change (%)"},
        yaxis2={"title": "Avg. Sentiment Score", "overlaying": "y", "side": "right"},
        hovermode="x"
    )
    st.plotly_chart(fig_industry)
else:
    st.warning("🔍 Select at least one industry to display the analysis.")

# 🔍 **Correlation Analysis**
st.subheader("🔬 Correlation Analysis (Stock % Change vs. Sentiment Score)")
pearson_corr = df[["Stock_Change%", "Sentiment_Score"]].corr(method="pearson").iloc[0, 1]
spearman_corr = df[["Stock_Change%", "Sentiment_Score"]].corr(method="spearman").iloc[0, 1]

st.metric(label="Pearson Correlation", value=round(pearson_corr, 3))
st.metric(label="Spearman Correlation", value=round(spearman_corr, 3))
#=============================================================================

#============================================================================
st.subheader("📈 Predict Future Stock Price")

# Load the saved model
model = load_model("Store/sp500_lstm_model_with_sent.h5")

# Load the data for prediction
predict_df = pd.read_csv("Store/final_dataset.csv")
predict_df['Date'] = pd.to_datetime(predict_df['Date'])
predict_df = predict_df.sort_values('Date')

features = ['S&P500', 'MEDCPIM158SFRBCLE', 'EFFR', 'FinBERT_Score']
scaler = MinMaxScaler()
predict_df[features] = scaler.fit_transform(predict_df[features])
data_values = predict_df.drop('Date', axis=1).values

# Prepare input sequence (last 12 months)
time_window = 12
input_seq = data_values[-time_window:]
input_seq = np.expand_dims(input_seq, axis=0)  # reshape to match LSTM input

# Predict
prediction = model.predict(input_seq)

# Inverse transformation to get the actual predicted value
dummy_pred = np.zeros((1, data_values.shape[1]))
dummy_pred[0, 0] = prediction
actual_prediction = scaler.inverse_transform(dummy_pred)[0, 0]

# Calculate next prediction date
latest_date = predict_df['Date'].max()
next_month_date = latest_date + pd.DateOffset(months=1)
prediction_month_year = next_month_date.strftime("%B %Y")

# Display the predicted stock price
st.success(f"### Predicted S&P500 Price for {prediction_month_year}: ${actual_prediction:.2f}")

# Visualize the recent trend including prediction
recent_actual = scaler.inverse_transform(data_values[-time_window:, :])[:, 0]

# Create X-axis labels including prediction month
date_labels = [date.strftime('%b %Y') for date in predict_df['Date'].iloc[-time_window:]] + [prediction_month_year]

fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=date_labels[:-1], y=recent_actual,
                              mode='lines+markers', name='Recent Actual Prices', line=dict(color="blue")))
fig_pred.add_trace(go.Scatter(x=[date_labels[-2], date_labels[-1]],
                              y=[recent_actual[-1], actual_prediction],
                              mode='lines+markers', name='Prediction', line=dict(color="red", dash="dash")))

fig_pred.update_layout(title='Recent Actual Prices and Prediction',
                       xaxis_title='Month',
                       yaxis_title='S&P500 Price',
                       hovermode='x')
st.plotly_chart(fig_pred)
#========================================================================
# 📌 **Generate QR Code for Sharing**
st.subheader("📱 Share this App!")
app_url = "https://your-app-link.com"  # Replace with your actual deployment link
qr = qrcode.make(app_url)
qr_img = BytesIO()
qr.save(qr_img, format="PNG")
st.image(qr_img.getvalue(), caption="Scan this QR Code to access the app")

# 📌 Footer - Ownership Information
st.markdown("---")  # Adds a horizontal line for separation
st.markdown("""
#### 📌 Developed By:
👨‍💻 **Million Solomon**  
👨‍💻 **Safwan Hasan**  

This web app was created as part of our research on **AI-driven financial sentiment analysis and stock movement trends**.  
""")

st.markdown("📅 **Last Updated:** March 2025")
st.markdown("🔗 **For inquiries, contact us at:** [million.solomon@torontomu.ca](mailto:million.solomon@torontomu.ca), [safwan.hasan@torontomu.ca](mailto:safwan.hasan@torontomu.ca)")

#==================================================================