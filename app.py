import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
# import pandas_datareader as pdr
from keras.models import load_model
import streamlit as  st

start = '2014-01-01'
from datetime import date

# Get today's date
end = date.today()


st.title("Stock Trend Prediction📈")


user_input=st.text_input("Enter input:")

try:
    df = yf.download(user_input, start=start, end=end)
except Exception as e:
    print(f"An error occurred: {e}")

#Describing data

st.subheader("Data from 2014 - 2024")
st.write(df.describe())

# Visualizations

st.subheader("Closing Price VS Time")
fig=plt.figure(figsize=(12,6))
plt.plot(df['Close'])
st.pyplot(fig)


st.subheader("Closing Price VS Time with MA")
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df['Close'])
st.pyplot(fig)


st.subheader("Closing Price VS Time with 100MA and 200MA")

# Calculate moving averages
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

# Create a figure
fig = plt.figure(figsize=(12, 6))

# Plot the moving averages and closing price
plt.plot(ma100, 'r', label='100-Day MA')  # Red line for 100-day moving average
plt.plot(ma200, 'g', label='200-Day MA')  # Green line for 200-day moving average
plt.plot(df['Close'], 'b', label='Closing Price')  # Blue line for closing price

# Adding labels and title
plt.title("Closing Price and Moving Averages")
plt.xlabel("Time")  # X-axis label
plt.ylabel("Price")  # Y-axis label

# Adding a legend
plt.legend()

# Display the plot in Streamlit
st.pyplot(fig)


#Splitting training and test data
train_data=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
test_data=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
print(train_data,test_data)


#Rescaling your features
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))


data_training_array=scaler.fit_transform(train_data)

#Load the model

model=load_model('my_model.keras')

#Testing part

past_100_days=train_data.tail(100)
final_df=pd.concat([past_100_days,test_data],ignore_index=True)

input_data=scaler.fit_transform(final_df)



x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

scaler=scaler.scale_
scale_factor=1/scaler[0]
y_pred= model.predict(x_test)


y_pred=y_pred*scale_factor
y_test=y_test*scale_factor


#final output

st.subheader("Actual V/S Predicted Closing price")
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,color='b',label='Actal Close Price')
plt.plot(y_pred,color='r',label='Predicted Close Price')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)


# Add a title using Markdown
st.write("### Model Evaluation Results")
results = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
st.dataframe(results)


from sklearn.metrics import mean_absolute_error

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
st.subheader("Mean Absolute Error (MAE)")
st.write(mae)

from sklearn.metrics import mean_squared_error
# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
st.subheader("Mean Squared Error (MSE)")
st.write(mse)

from sklearn.metrics import r2_score

# Calculate R-squared Score
r2 = r2_score(y_test, y_pred)
st.subheader("R-squared Score")
st.write(r2)


# Calculate accuracy percentage
accuracy_percentage = (1 - (mae / np.mean(y_test))) * 100
st.subheader("Model Accuracy Percentage")
st.write(f"{accuracy_percentage:.2f}%")

