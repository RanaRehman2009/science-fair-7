import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go 
'''
Here I'm just importing modules for the rest of the project.
'''
#------------------------------------------- 

start = '2021-01-01'
today = date.today().strftime("%Y-%m-%d")

st.title("Science Fair")
stock_options = ('AAPL', 'GOOG', "MSFT", "GME")
selected_stock = st.selectbox("Select stock for prediction", stock_options)
n_years = st.slider("Number of days",1, 365)
perdiod = n_years

@st.cache
def load_data(ticker):
  data = yf.download(ticker, start, today)
  data.reset_index(inplace=True)
  return data

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Data Done")

st.subheader("Raw dataset")
st.write(data.tail())

def plot_raw_data():
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open'))
  fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close'))
  fig.layout.update(title_text='Current Data', xaxis_rangeslider_visible=True)
  st.plotly_chart(fig)
plot_raw_data()
# Forecast
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=perdiod)
forecast = m.predict(future)

st.subheader('Forecast Data')
st.write(forecast.tail())

st.write('Forecast Data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast Parts')
fig2 = m.plot_components(forecast)
st.write(fig2)

