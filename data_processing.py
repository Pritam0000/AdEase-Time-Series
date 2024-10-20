import pandas as pd
import numpy as np

def load_and_process_data():
    # Load the main data
    df = pd.read_csv('train_1.csv')

    # Melt the dataframe to long format
    df = df.melt(id_vars=['Page'], var_name='date', value_name='views')

    # Parse the page name
    df[['title', 'domain', 'access_type', 'access_origin']] = df['Page'].str.split('_', expand=True)
    df['language'] = df['domain'].str.split('.', expand=True)[0]

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Sort the dataframe
    df = df.sort_values(['Page', 'date'])

    # Load the exogenous campaign data
    with open('Exog_Campaign_eng', 'r') as file:
        exog_data = [int(line.strip()) for line in file]

    # Create a date range for the exogenous data
    exog_dates = pd.date_range(start=df['date'].min(), periods=len(exog_data), freq='D')

    # Create a DataFrame for the exogenous data
    exog_campaign = pd.DataFrame({
        'date': exog_dates,
        'campaign': exog_data
    })

    return df, exog_campaign

def prepare_data_for_modeling(df, language, exog_campaign):
    # Filter data for the selected language
    df_lang = df[df['language'] == language].groupby('date')['views'].mean().reset_index()
    
    # Merge with exogenous data
    df_merged = pd.merge(df_lang, exog_campaign, on='date', how='left')
    
    # Fill any missing values in the campaign column with 0
    df_merged['campaign'] = df_merged['campaign'].fillna(0)
    
    return df_merged