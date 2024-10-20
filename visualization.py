import plotly.express as px

def plot_time_series(df, language):
    df_lang = df[df['language'] == language]
    fig = px.line(df_lang, x='date', y='views', color='Page', title=f'Page Views Over Time - {language}')
    return fig

def plot_language_distribution(df):
    lang_dist = df.groupby('language')['views'].sum().sort_values(ascending=False)
    fig = px.bar(lang_dist, x=lang_dist.index, y='views', title='Total Views by Language')
    return fig