import streamlit as st
import pandas as pd
import plotly.express as px
from data_processing import load_and_process_data
from visualization import plot_time_series, plot_language_distribution
from modeling import train_and_forecast_arima, train_and_forecast_prophet

def main():
    st.title("AdEase Time Series Analysis and Forecasting")

    # Load and process data
    df, exog_campaign = load_and_process_data()

    # Sidebar for user input
    st.sidebar.header("Settings")
    selected_language = st.sidebar.selectbox("Select Language", df['language'].unique())
    forecast_days = st.sidebar.slider("Forecast Days", 1, 30, 7)

    # Display basic stats
    st.header("Data Overview")
    st.write(f"Total pages: {df['Page'].nunique()}")
    st.write(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Visualizations
    st.header("Visualizations")
    fig_time_series = plot_time_series(df, selected_language)
    st.plotly_chart(fig_time_series)

    fig_lang_dist = plot_language_distribution(df)
    st.plotly_chart(fig_lang_dist)

    # Modeling
    st.header("Time Series Forecasting")
    model_choice = st.radio("Select Model", ["ARIMA", "Prophet"])

    if model_choice == "ARIMA":
        forecast, mape = train_and_forecast_arima(df, selected_language, forecast_days, exog_campaign)
    else:
        forecast, mape = train_and_forecast_prophet(df, selected_language, forecast_days)

    st.write(f"MAPE: {mape:.2f}%")
    st.line_chart(forecast)

if __name__ == "__main__":
    main()