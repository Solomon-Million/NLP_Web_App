## 📌 README.md

### Project Title:
**Stock & Sentiment Analysis Web Application**

### Developed By:
- **Million Solomon** ([million.solomon@torontomu.ca](mailto:million.solomon@torontomu.ca))  
- **Safwan Hasan** ([safwan.hasan@torontomu.ca](mailto:safwan.hasan@torontomu.ca))

---

## 🚀 Project Description:
This project is a web-based analytical tool designed to explore relationships between textual sentiment from financial disclosures (10-K risk factors) and stock market performance of S&P 500 companies. Additionally, the application predicts future stock prices based on historical financial indicators and sentiment analysis using machine learning (LSTM neural network).

---

## 🎯 Main Objectives:
- Analyze historical sentiment scores extracted from ITEM 1A (Risk Factors) in 10-K reports.
- Visualize and compare sentiment trends with actual historical stock price movements.
- Identify correlations between financial sentiment and stock price movements.
- Predict future stock prices leveraging LSTM-based deep learning models.

---

## ⚙️ Technologies Used:
- **Python**  
- **Streamlit** (Web Application Interface)
- **TensorFlow/Keras** (LSTM Prediction Model)
- **Pandas & NumPy** (Data Manipulation)
- **scikit-learn** (Data Preprocessing: MinMaxScaler)
- **Plotly** (Interactive Visualizations)
- **QR Code** (For sharing web app)

---

## 📂 Folder Structure:
```
Submission/
├── app.py                                 # Streamlit Web App (main script)
├── Store/
│   ├── merged_data_final.csv              # Main dataset for company analysis
│   ├── final_dataset.csv                  # Dataset used for stock price prediction
│   └── sp500_lstm_model_with_sent.h5      # Pre-trained LSTM model for prediction
├── requirements.txt                       # Python libraries required to run the app
└── README.md                              # This README file
```

---

## 📌 About `app.py`:

`app.py` is a Python Streamlit-based web application that provides users an interactive platform for exploring stock and sentiment trends of companies listed in the S&P 500. It contains the following sections and functionalities:

### 1. 📊 Stock & Sentiment Analysis:
- Company-level visualization of historical stock percentage changes alongside sentiment trends.
- Sector and industry-level analyses with selectable views to identify broad market trends.
- Pearson and Spearman correlation analysis between sentiment scores and stock movements.

### 2. 📈 Stock Price Prediction:
- Predicts the next month's S&P500 price using a trained Long Short-Term Memory (LSTM) neural network model (`sp500_lstm_model_with_sent.h5`).
- Utilizes recent 12 months' historical financial data (`final_dataset.csv`) as input for prediction.
- Clearly indicates the prediction month and year.
- Displays recent actual prices and predicted prices on an interactive graph.

### 3. 📱 QR Code Generation:
- Generates a QR code to facilitate easy sharing of the web application link.

### 4. 🖥️ Running Instructions:
Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📌 Dependencies (`requirements.txt`):
```
streamlit
numpy
pandas
plotly
tensorflow
scikit-learn
```

---

## 📌 Contact Information:
For inquiries or additional information, please contact us at:  
- **Million Solomon**: [million.solomon@torontomu.ca](mailto:million.solomon@torontomu.ca)  
- **Safwan Hasan**: [safwan.hasan@torontomu.ca](mailto:safwan.hasan@torontomu.ca)

---

## 📅 Last Updated:
**March 2025**

