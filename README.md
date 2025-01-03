
# 📊 **Stock Trend Prediction Using Deep Learning and Streamlit**

## 🚀 **Project Overview**  
This project predicts stock market trends using historical stock data and deep learning techniques. It utilizes a pre-trained Keras model to forecast stock prices and visualizes results interactively via a Streamlit web application.

## 🛠️ **Tools & Technologies Used**  
- **Python**  
- **Streamlit** – For creating an interactive web application  
- **Keras** – For building and loading the deep learning model  
- **yFinance** – For fetching stock market data  
- **NumPy & Pandas** – For data manipulation and analysis  
- **Matplotlib & Seaborn** – For data visualization  
- **Scikit-learn** – For data preprocessing and evaluation metrics  

## 📈 **Key Features**  
1. **Stock Data Fetching:** Real-time stock data is fetched using `yfinance`.  
2. **Interactive Visualizations:** Users can visualize:  
   - Closing Price vs Time  
   - Moving Averages (100-day & 200-day)  
3. **Deep Learning Model Integration:** A pre-trained LSTM model predicts stock prices.  
4. **Model Evaluation:** Metrics like MAE, MSE, R² Score, and Accuracy Percentage are displayed.  
5. **Dynamic User Input:** Users can input stock tickers for analysis.  

## 🎯 **How to Run the Project?**  
1. Clone the repository:  
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:  
   ```bash
   streamlit run app.py
   ```
4. Open the app in your browser at `http://localhost:8501`

## 📊 **Example Usage**  
- Enter a valid stock ticker symbol (e.g., `AAPL` for Apple).  
- Explore time series visualizations.  
- Compare predicted vs actual closing prices.  

## 🧠 **Evaluation Metrics**  
- **Mean Absolute Error (MAE)**  
- **Mean Squared Error (MSE)**  
- **R-squared Score (R²)**  
- **Accuracy Percentage**

## 🤝 **Contributing**  
Feel free to fork the repository and submit pull requests. Contributions are welcome!  


---

Let me know if you want me to add installation instructions for a specific environment or additional details! 😊
