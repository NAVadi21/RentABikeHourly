import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


#Heading of the project
st.markdown(
    f"""
    <h1 style='text-align: center; color: blue; font-size: 3.2rem;'>{'HOURLY BIKE RENTAL DEMAND ANALYSER &#x1F6B2;'}</h1>
    """,
    unsafe_allow_html=True
)
st.title("")

# uploaded_file1=st.sidebar.file_uploader("Choose a train file to upload")
# uploaded_file2=st.sidebar.file_uploader("Choose a test file to upload")

#Loading data sets which are in the form of csv files
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# Display Train data
st.markdown(
    f"""
    <h1 style='color: green; font-size: 2rem;'>{'Train Data:'}</h1>
    """,
    unsafe_allow_html=True
)
st.write(train_data.head())

# Display Test data
st.markdown(
    f"""
    <h1 style='color: green; font-size: 2rem; '>{'Test Data:'}</h1>
    """,
    unsafe_allow_html=True
)
st.write(test_data.head())

# Display basic statistics
st.markdown(
    f"""
    <h1 style='color: green; font-size: 2rem; '>{'Train Data Statistics:'}</h1>
    """,
    unsafe_allow_html=True
)
# st.header('Train Data Statistics:')
st.write(train_data.describe())


# Looking at the correlation between numerical variables
st.markdown(
    f"""
    <h1 style='color: green ; font-size: 2rem; ;'>{'Heatmap of factors affecting rental bike demand:'}</h1>
    """,
    unsafe_allow_html=True
)
corr = train_data[["temp","atemp","casual","registered","humidity","windspeed","count"]].corr()
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
ax=sn.heatmap(corr, mask=mask,vmax=.9, square=True,annot=True, cmap="YlGnBu")
st.pyplot(fig)

# Visualize bike rental demand over time
st.markdown(
    f"""
    <h1 style='color: green ; font-size: 2rem; '>{'Visualize bike rental demand over time(1st month):'}</h1>
    """,
    unsafe_allow_html=True
)
train_data['datetime'] = pd.to_datetime(train_data['datetime'])
fig, ax = plt.subplots()
ax.plot(train_data.head(30)['datetime'], train_data.head(30)['count'])
plt.xticks(rotation='vertical')
ax.set_xlabel('Date')
ax.set_ylabel('Hourly bike rental demand')
st.pyplot(fig)


# extracting date, hour and month from the datetime
train_data['datetime'] = pd.to_datetime(train_data['datetime'], format='%Y-%m-%d %H:%M:%S')

train_data['hour'] = train_data['datetime'].dt.hour
train_data['dayofweek'] = (train_data['datetime'].dt.day)%7


test_data['datetime'] = pd.to_datetime(test_data['datetime'], format='%Y-%m-%d %H:%M:%S')

test_data['hour'] = test_data['datetime'].dt.hour
test_data['dayofweek'] = (test_data['datetime'].dt.day)%7


#Displaying updated data
st.markdown(
    f"""
    <h1 style='color: green ; font-size: 2rem;'>{'Updated Train Data:'}</h1>
    """,
    unsafe_allow_html=True
)
st.write(train_data.head())

st.markdown(
    f"""
    <h1 style='color: green ; font-size: 2rem; '>{'Updated Test Data:'}</h1>
    """,
    unsafe_allow_html=True
)
st.write(test_data.head())


# Build and display predictive model
features = ['temp', 'atemp', 'humidity', 'windspeed', 'hour', 'dayofweek']
X_train = train_data[features]
y_train = train_data['count']
X_test = test_data[features]

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_train)
# st.markdown(
#     f"""
#     <h1 style='color: green ; font-size: 2rem; '>{'R^2 score Calculation:'}</h1>
#     """,
#     unsafe_allow_html=True
# )
# st.write('R^2 score:', r2_score(y_train, y_pred))

# Add interactive elements
st.markdown(
    f"""
    <h1 style='color: green ; font-size: 2rem; '>{'Prediction using interactive slider:'}</h1>
    """,
    unsafe_allow_html=True
)
hour = st.slider('Hour of the day', 0, 23, 12)
temp = st.slider('Temperature', -20, 50, 20)
atemp = st.slider('Feels like temperature', -20, 50, 20)
hum = st.slider('Humidity', 0, 100, 50)
windspeed = st.slider('Windspeed', 0, 50, 25)
dayofweek = st.slider('Day of Week', 0, 6, 3)

input_data = [[temp, atemp, hum, windspeed, hour, dayofweek]]
prediction = model.predict(input_data)

st.write('Predicted bike rentals:', int(prediction))
