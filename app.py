# Import libraries
import pandas as pd
import numpy as np
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation
import plotly.graph_objects as go
from datetime import timedelta
from sklearn.metrics import r2_score
import streamlit as st

#Creates a wide layout, page title and configuration for the Streamlit app
st.set_page_config(
    page_title="Electricity Demand Forecaster",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded")

#Creates a title
st.title('Great Britain Hourly Electricity Demand Forecaster 🔋⚡')

with st.sidebar:
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/nathan-rodrigues/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Nathan Rodrigues`</a>', unsafe_allow_html=True)
    #Provides a description on the app and how to use it
    st.caption(
        f"This tool uses the Prophet forecasting package developed by Meta (link below) to forecast hourly load values (MW) in Great Britain - based solely on historical data. Play around with the hyperparameters below to discover how they affect the model accuracy."
        )
    prophet_url = "https://facebook.github.io/prophet/"
    st.markdown(f'<a href="{prophet_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://uxwing.com/wp-content/themes/uxwing/download/brands-and-social-media/meta-icon.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Prophet | Forecasting at scale`</a>', unsafe_allow_html=True)
    st.subheader("Choose Your :red[Parameters] 🎯")
    n = st.slider('Horizon (hours)', min_value=1, max_value=1440, value=24, step=1)
    cp = st.slider('Changepoint Prior Scale', min_value=0.001, max_value=0.5, value=0.05, step=0.001)
    sp = st.slider('Seasonality Prior Scale', min_value=0.01, max_value=10.0, value=10.0, step=0.01)
    type = st.selectbox("Choose a Seasonality Mode:", ("additive", "multiplicative"))
    ds = st.checkbox("Daily Seasonality", value=True)
    ys = st.checkbox("Yearly Seasonality", value=True)
    st.caption('**Source**: _European Network of Transmission System Operators for Electricity_')
    source_url = "https://www.entsoe.eu/data/power-stats/"
    st.markdown(f'<a href="{source_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ8b6pcuOYx5_iHQ7iSmvbXqh0Hy5RY6JuQNH4c9qBsWqBtn0X0BborYQRolvNpTXY4J58&usqp=CAU" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`ENTSO-E`</a>', unsafe_allow_html=True)

#####################################
# Load data for Great Britain
GB = pd.read_excel('GB.xlsx')

# Model and Prediction
m = Prophet(changepoint_prior_scale = cp, 
            seasonality_prior_scale = sp,
            seasonality_mode = type,
            daily_seasonality=ds,
            yearly_seasonality=ys
            ).fit(GB)
future = m.make_future_dataframe(n, freq='h')
fcst = m.predict(future)

# Plot the forecasts
fig = plot_plotly(m, fcst)

# Customize layout for better visualization
fig.update_layout(
    xaxis_title = "Date",
    yaxis_title = "Load Values (MW)",
    template = "plotly_dark",  # Dark theme
    yaxis=dict(
        autorange=True,
        fixedrange=False
    ),
    xaxis=dict(
        rangeselector=dict(visible=True)  # Disable timeframe buttons
    )
)

# Update scatter plot markers for demand values to be white
fig.update_traces(
    marker=dict(color="mintcream"),
    selector=dict(mode="markers", type="scatter")
)

# Compute the R-squared
metric_df = fcst.set_index('ds')[['yhat']].join(GB.set_index('ds').y).reset_index()
metric_df.dropna(inplace = True)
r2 = r2_score(metric_df.y, metric_df.yhat)
r2_formatted = '{:,.2%}'.format(r2)

# Average hourly forecast
abs = fcst.tail(n)
abs_avg = np.average(abs["yhat"])
abs_avg_formatted = '{:,.2f} MW'.format(abs_avg)

with st.container(border = True):
    st.subheader("**R-Squared**")
    st.metric("**R-Squared**", r2_formatted, delta=None, delta_color="normal", help=None, label_visibility="collapsed",use_container_width=True)
with st.container(border = True):
    st.subheader(f"**Average hourly forecast for the next :red[{n} hours] is**")
    st.metric(f"**Average forecast for the next :red[_{n} hours_] is**", abs_avg_formatted, delta=None, delta_color="normal", help=None, label_visibility="collapsed", use_container_width=True)

st.markdown(f'👇 Hover over the interactive chart to zoom in or out 🔍')

with st.container(border = True):
    # Display the figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)

#Formats the title to be placed higher on the app
st.markdown(
    """
        <style>
            .appview-container .main .block-container {{
                padding-top: {padding_top}rem;
                padding-bottom: {padding_bottom}rem;
                }}

        </style>""".format(
        padding_top=2, padding_bottom=2
    ),
    unsafe_allow_html=True,
)
