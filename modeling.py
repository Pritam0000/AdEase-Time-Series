import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
from data_processing import prepare_data_for_modeling

def train_and_forecast_arima(df, language, forecast_days, exog_campaign):
    df_merged = prepare_data_for_modeling(df, language, exog_campaign)
    
    # Train-test split
    train_size = int(len(df_merged) * 0.8)
    train, test = df_merged[:train_size], df_merged[train_size:]
    
    # Fit ARIMA model
    model = ARIMA(train['views'], exog=train['campaign'], order=(1, 1, 1))
    results = model.fit()
    
    # Forecast
    forecast = results.forecast(steps=len(test), exog=test['campaign'])
    
    # Calculate MAPE
    mape = mean_absolute_percentage_error(test['views'], forecast)
    
    # Forecast future
    future_exog = pd.DataFrame(index=pd.date_range(start=df_merged['date'].max() + pd.Timedelta(days=1), periods=forecast_days))
    future_exog['campaign'] = 0  # Assume no campaign in the future
    future_forecast = results.forecast(steps=forecast_days, exog=future_exog['campaign'])
    
    return pd.concat([df_merged.set_index('date')['views'], future_forecast]), mape * 100

def train_and_forecast_prophet(df, language, forecast_days):
    df_merged = prepare_data_for_modeling(df, language,exog_campaign)
    df_merged.columns = ['ds', 'y', 'campaign']
    
    # Train-test split
    train_size = int(len(df_merged) * 0.8)
    train, test = df_merged[:train_size], df_merged[train_size:]
    
    # Fit Prophet model
    model = Prophet()
    model.add_regressor('campaign')
    model.fit(train)
    
    # Forecast
    future = model.make_future_dataframe(periods=len(test))
    future['campaign'] = df_merged['campaign']
    forecast = model.predict(future)
    
    # Calculate MAPE
    mape = mean_absolute_percentage_error(test['y'], forecast['yhat'][-len(test):])
    
    # Forecast future
    future = model.make_future_dataframe(periods=forecast_days)
    future['campaign'] = 0  # Assume no campaign in the future
    future_forecast = model.predict(future)
    
    return pd.DataFrame({'date': future_forecast['ds'], 'views': future_forecast['yhat']}), mape * 100