# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # This is a test to see how to make the widgets work.
#
#

# %%
import os
import warnings
import pandas as pd
warnings.simplefilter(action='ignore', category=UserWarning)

# %% [markdown]
# ## Trying to create the widgets from there:

# %% [markdown]
# ## !! WARNING I LOAD THE SHITTY ZURICH DF, WILL PROBABLY NEED TO CHANGE IT!!!!

# %%
zurich_stops_df=pd.read_csv('zurich.csv')
zurich_stops_df.drop(columns=zurich_stops_df.columns[0], axis=1, inplace=True)
zurich_stops_df =zurich_stops_df.rename(columns={'allstops.stop_id': 'stop_id', 'allstops.stop_name': 'stop_name',
                               'allstops.latitude':'latitude', 'allstops.longitude':'longitude',
                               'allstops.location_type':'location_type','allstops.parent_station':'parent_station'})
zurich_stops_df.head()


# %%
def unique_sorted_values(array):
    unique = array.unique().tolist()
    unique.sort()
    return unique


# %%
list_color=['#636EFA', # the plotly blue you can see above
 '#EF553B',
 '#00CC96',
 '#AB63FA',
 '#FFA15A',
 '#19D3F3',
 '#FF6692',
 '#B6E880',
 '#FF97FF',
 '#FECB52']

# %%
import plotly.graph_objects as go
# Function definition for plotting the travel
def plot_map(journey_df):
    fig = go.Figure()
   
    for i in range(len(journey_df)):
        # Below for segment between stations
        fig.add_trace(go.Scattermapbox(
            #mode = "markers",
            mode= "lines",
            lat = journey_df.iloc[i:i+2]['latitude'].tolist(),
            lon = journey_df.iloc[i:i+2]['longitude'].tolist(),
            hovertext = journey_df['stop_name'].tolist(),
            marker = {'color': list_color[i], 
                      "size": 10},
        ))
        # Below for dot at stations
        fig.add_trace(go.Scattermapbox(
            mode = "markers",
            lat = journey_df['latitude'].tolist(),
            lon = journey_df['longitude'].tolist(),
            hovertext = journey_df['stop_name'].tolist(),
            marker = {'color': 'black', 
                      "size": 10},
        ))
        
    # To display the maps at the good location
    mean_lat=journey_df['latitude'].mean()
    mean_lon=journey_df['longitude'].mean()
    fig.update_layout(margin ={'l':0,'t':0,'b':0,'r':0},
                      mapbox = {
                          'center': {'lat': mean_lat,'lon': mean_lon},
                          'style': "stamen-terrain",
                          'zoom': 9},
                      width=800,
                      height=300,)
    fig.show()


# %%
from ipywidgets import HBox, VBox, Layout,Button
import ipywidgets as widgets
import numpy as np
from IPython.display import display, clear_output
import datetime

output = widgets.Output()

#Start destination
start_w = widgets.Dropdown(
    options=unique_sorted_values(zurich_stops_df['stop_name']),
    #options=zurich_stops_df.sort_values(by=['stop_name'])['stop_name'],#unique_sorted_values(stops_zurich['allstops.stop_name']),
    value='Zürich HB',
    description='From:')
#Stop destination
stop_w = widgets.Dropdown(
    options=unique_sorted_values(zurich_stops_df['stop_name']),
    #options=zurich_stops_df.sort_values(by=['stop_name'])['stop_name'],#unique_sorted_values(stops_zurich['allstops.stop_name']),
    value='Zürich HB',
    description='To:')
#Date Picker
date_w = widgets.DatePicker(description='Pick a Date',value=datetime.date.today(),)
#Hour selector
hour_w = widgets.Dropdown(
    options=np.arange(24),
    value=0,
    description='H:',
    layout=Layout(width='140px'))
#Minute selector
min_w = widgets.Dropdown(
    options=np.arange(60),
    value=0,
    description='M:',
    layout=Layout(width='140px'))

#slider for uncertainty
confidence_w = widgets.IntSlider(
    value=100,
    min=0,
    max=100,
    step=1,
    description='Confidence:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)
#Enter button
enter_b = Button(description='Enter',
           layout=Layout(width='300px'))
enter_b.style.button_color = 'lightblue'
#Submodule for hour and minutes
time_items=[hour_w,min_w]
sub_box = HBox(children=time_items)
#Overall widget box
items = [start_w,stop_w,date_w,sub_box, confidence_w]
big_box = VBox(children=items) # Vertical display

display(big_box) 
display(enter_b, output)

def on_button_clicked(b):
    with output:
        clear_output()
        print('You have selected for start destination:', start_w.value)
        print('Which corresponds to stop_id:', \
              zurich_stops_df.loc[zurich_stops_df['stop_name'] == start_w.value, 'stop_id'].values[0])
        print('And for end destination:', stop_w.value)
        print('Which corresponds to stop_id:', \
              zurich_stops_df.loc[zurich_stops_df['stop_name'] == stop_w.value, 'stop_id'].values[0])
        # Get the day of the week
        day_of_week = date_w.value.strftime('%A')
        print('On a :', day_of_week)
        print('With a confidence of :', confidence_w.value)
        print('WARNING! AS IT IS NOW THE JOURNEY SHOWN ON THE MAP HAS NOTHING TO DO WITH THE ACTUAL INPUT DESTINATIONS')
        
        ## IN HERE WE NEED TO PUT THE STOPS IN THE ORDER THAT WE WANT TO MAKE
        ## THE JOURNEY! THEN THE PLOT_MAP FCT DO THE REST
        row_1 =zurich_stops_df.loc[zurich_stops_df['stop_name'] == start_w.value].iloc[0:1]
        row_2=zurich_stops_df.loc[zurich_stops_df['stop_name'] == stop_w.value].iloc[0:1]
        journey_df=pd.concat([row_1,row_2])
        display(journey_df)
        #journey_df=stops_zurich.tail(5) #
        plot_map(journey_df)

enter_b.on_click(on_button_clicked)

# %% [markdown]
# ## Doing it the other way around for the P

# %%
from ipywidgets import HBox, VBox, Layout,Button
import ipywidgets as widgets
import numpy as np
from IPython.display import display, clear_output
import datetime

output = widgets.Output()

#Start destination
start_w = widgets.Dropdown(
    options=zurich_stops_df['stop_id'],#unique_sorted_values(stops_zurich['allstops.stop_name']),
    #value='Zürich HB',
    description='From:')
#Stop destination
stop_w = widgets.Dropdown(
    options=zurich_stops_df['stop_id'],#unique_sorted_values(stops_zurich['allstops.stop_name']),
    #value='Zürich HB',
    description='To:')
#Date Picker
date_w = widgets.DatePicker(description='Pick a Date',value=datetime.date.today(),)
#Hour selector
hour_w = widgets.Dropdown(
    options=np.arange(24),
    value=0,
    description='H:',
    layout=Layout(width='140px'))
#Minute selector
min_w = widgets.Dropdown(
    options=np.arange(60),
    value=0,
    description='M:',
    layout=Layout(width='140px'))
#Enter button
enter_b = Button(description='Enter',
           layout=Layout(width='300px'))
enter_b.style.button_color = 'lightblue'
#Submodule for hour and minutes
time_items=[hour_w,min_w]
sub_box = HBox(children=time_items)
#Overall widget box
items = [start_w,stop_w,date_w,sub_box]
big_box = VBox(children=items) # Vertical display

display(big_box) 
display(enter_b, output)

def on_button_clicked(b):
    with output:
        clear_output()
        print('You have selected for start destination:', start_w.value)
        print('Which corresponds to stop_id:', \
              zurich_stops_df.loc[zurich_stops_df['stop_id'] == start_w.value, 'stop_name'].values[0])
        print('And for end destination:', stop_w.value)
        print('Which corresponds to stop_id:', \
              zurich_stops_df.loc[zurich_stops_df['stop_id'] == stop_w.value, 'stop_name'].values[0])
        # Get the day of the week
        day_of_week = date_w.value.strftime('%A')
        print('On a :', day_of_week)
        print('WARNING! AS IT IS NOW THE JOURNEY SHOWN ON THE MAP HAS NOTHING TO DO WITH THE ACTUAL INPUT DESTINATIONS')
        
        ## IN HERE WE NEED TO PUT THE STOPS IN THE ORDER THAT WE WANT TO MAKE
        ## THE JOURNEY! THEN THE PLOT_MAP FCT DO THE REST
        row_1 =zurich_stops_df.loc[zurich_stops_df['stop_id'] == start_w.value]
        row_2=zurich_stops_df.loc[zurich_stops_df['stop_id'] == stop_w.value]
        journey_df=pd.concat([row_1,row_2])
        #journey_df=stops_zurich.tail(5) #
        display(journey_df.head())
        plot_map(journey_df)

enter_b.on_click(on_button_clicked)

# %%
start_id=8591233
zurich_stops_df[zurich_stops_df['stop_name']=='Zürich Flughafen, Bahnhof']

# %%
