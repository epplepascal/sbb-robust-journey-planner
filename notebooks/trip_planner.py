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
# # Journey Planner
#
# NOTE: the notebook needs to be run with maximum specs in order for the kernel not to die
#
# This file runs our implementation of CSA and computes the probability of success of the found trips. The user interface is launch in the last cell of the script

# %%
import os
import warnings
import pandas as pd
from collections import namedtuple
import re
import pickle
import time
warnings.simplefilter(action='ignore', category=UserWarning)

# %%
# !git lfs pull

# %% [markdown]
# ## CSA setup

# %% [markdown]
# Pre-processed timetable file loading

# %%
Connection = namedtuple('Connection', ['trip_id', 'dep_stop', 'arr_stop', 'dep_time_sec', 'arr_time_sec', 'route_desc', 'active_days', 'active'])
with open('../data/ls_timetable.pkl', 'rb') as f:
    ls_timetable = pickle.load(f)
original_length = len(ls_timetable)

# %% [markdown]
# Loading file with walking times between stations. Each row is made up of pairs of stations that are maximum 500m from each other

# %%
transfer_times = pd.read_csv('../data/df_with_transfer_time.csv')
transfer_times['stop_id1'] = transfer_times['stop_id1'].astype('string')
transfer_times['stop_id2'] = transfer_times['stop_id2'].astype('string')

# %% [markdown]
# Loading file with stations within a 15km radius from Zurich central station

# %%
with open('../data/stations.pkl', 'rb') as f:
    stations = pickle.load(f)

# %% [markdown]
# Creating dictionary with all the stations reachable on foot starting from a certain station

# %%
transfer_dict = {}
for _, row in transfer_times.iterrows():
    if row.stop_id1 not in transfer_dict:
        transfer_dict[row.stop_id1] = [(row.stop_id2, row.walking_time)]
    elif (row.stop_id2, row.walking_time) not in transfer_dict[row.stop_id1]:
        transfer_dict[row.stop_id1].append((row.stop_id2, row.walking_time))


# %% [markdown]
# Utility function to reset timetable to orginal state (needed because of processing done inside CSA function). It essentially eliminates foothpaths added by CSA and reactivate all connections

# %%
def reset_timetable(t, original_length):
    for id, conn in enumerate(t):
        t[id] = conn._replace(active = 1)
    if(original_length!=len(t)):
        del t[original_length-len(t):]


# %% [markdown]
# ## CSA Algorithm
# CSA Algorithm (ref: https://github.com/trainline-eu/csa-challenge/blob/master/csa.py#L82, https://arxiv.org/pdf/1703.05997.pdf) with modifications to include foot paths and change of transport type

# %%
from array import array

### Classical CSA implementation
def CSA(stations, t, departure_station, departure_time, arrival_station, arrival_time, day):
    MAX_INT = 2**32 - 1

    # earliest arrival time in arrival_station
    earliest = MAX_INT
    # dictionary containing the earliest-arriving connection for each station
    in_connection = {s: MAX_INT for s in stations}
    # dictionary containing the the earliest arrival time possible for each station
    earliest_arrival = {s: MAX_INT for s in stations}
    
    earliest_arrival[departure_station] = departure_time
    
    # to handle case in which first connection has to be done on foot 
    if transfer_dict.get(departure_station) is not None:
        for reachable_station, transfer_time in transfer_dict[departure_station]:
            arrival_time_reachable = departure_time + transfer_time
            earliest_arrival[reachable_station] = arrival_time_reachable
            walking_connection = Connection('W', departure_station, reachable_station, departure_time, arrival_time_reachable, 'W', [day], 1)
            t.append(walking_connection)
            in_connection[reachable_station] = len(t) - 1 #get last index of t
    
    for i, c in enumerate(t):
        # change of transport type penalty in seconds (initially zero in case we're staying on the same trasport type)
        change = 0
        
        # check if connection is active (needed for Yen algorithm) and if it's the day that interests the user
        if (c.active == 1) and (day in c.active_days):
            # check if we have to apply change penalty
            if (in_connection.get(c.dep_stop) is not None): #sanity
                if (in_connection[c.dep_stop] != MAX_INT):
                    if (t[in_connection[c.dep_stop]].trip_id != c.trip_id):
                            change = 120 #transfer time penalty in seconds
            
            # check if we can catch the connection and if it improves what we already have
            if (c.dep_time_sec >= earliest_arrival[c.dep_stop] + change) & (c.arr_time_sec < earliest_arrival[c.arr_stop]):
                
                earliest_arrival[c.arr_stop] = c.arr_time_sec
                in_connection[c.arr_stop] = i

                if c.arr_stop == arrival_station:
                    earliest = min(earliest, c.arr_time_sec)

                # Updates foot paths for stations reachable by the current one 
                if transfer_dict.get(c.arr_stop) is not None: # to avoid inconsistency between the transfer table and timetable
                        for reachable_station, transfer_time in transfer_dict[c.arr_stop]:
                            arrival_time_reachable = c.arr_time_sec + transfer_time
                            if reachable_station in stations: # to avoid inconsistency between the transfer table and timetable 
                                if (arrival_time_reachable < earliest_arrival[reachable_station]):
                                    earliest_arrival[reachable_station] = arrival_time_reachable
                                    # Append the 'walking' as a new connection to the timetable (needed for path reconstruction)
                                    walking_connection = Connection('W', c.arr_stop, reachable_station, c.arr_time_sec, arrival_time_reachable, 'W', [day], 1)
                                    t.append(walking_connection)
                                    in_connection[reachable_station] = len(t) - 1 #get last index of t

                                    if reachable_station == arrival_station:
                                        earliest = min(earliest, arrival_time_reachable)

            # Avoid considering useless connections (connections in timetable are sorted by decreasing arrival time)
            elif c.arr_time_sec > earliest:
                break

    route = []
    # We have to rebuild the route from the arrival station
    last_connection_index = in_connection[arrival_station]

    while last_connection_index != MAX_INT:
        connection = t[last_connection_index]
        route.append(connection)
        last_connection_index = in_connection[connection.dep_stop]
    
    route=list(reversed(route))
    
    return route


# %% [markdown]
# ## Yen Algorithm setup

# %% [markdown]
# To extract the earliest arriving connection from a list of candidates

# %%
def extract_min(Candidates):
    best_arrival_time = float('inf')
    for idx, (journey, index) in enumerate(Candidates):
        arrival_time = journey[-1].arr_time_sec
        if arrival_time < best_arrival_time:
            best_arrival_time = arrival_time
            best_idx = idx
    best_candidate = Candidates[best_idx]
    Candidates.pop(best_idx)
    return best_candidate, Candidates


# %% [markdown]
# To get only "proper" connections (no foothpaths)

# %%
def get_connections(J):
    return [connection for connection in J if connection.trip_id != 'W']


# %% [markdown]
# To deactive connections that do not need to be considered

# %%
def update_timetable(timetable, S_pi, C_dev):
    for id, conn in enumerate(timetable):
        if conn.dep_stop in S_pi or conn.arr_stop in S_pi or conn in C_dev: 
            timetable[id] = conn._replace(active = 0) #update the timetable


# %% [markdown]
# ## Yen Algorithm
# Implements best-k path CSA (ref: https://hal.science/hal-03264788/document)

# %%
from itertools import chain

## Implementation for obtaining multiple trip proposition
def CSA_multiple(stations, t, departure_station, departure_time, arrival_station, arrival_time, day, k):
    # first CSA journey computation
    J_0 = CSA(stations, t, departure_station, departure_time, arrival_station, arrival_time, day)
    
    if len(J_0) != 0:
        Candidates = [(J_0, 0)] ## set it to 1
    else:
        raise('No path found')
    
    Output = []
    
    while len(Output) < k and len(Candidates) != 0:
        # reset timetable to original status
        reset_timetable(t, original_length)
        
        # extracts the path with the earliest arrival time alongside with its index
        (J, j), Candidates = extract_min(Candidates)
        moving_connections = get_connections(J)
        Output.append(J)
        
        # iterates through the connection of the path deviating each time less and less from the original 
        for i,connection in enumerate(moving_connections[j:]):
            reset_timetable(t, original_length)
            
            # list of connections of the original path that will be kept
            pi = []

            if i > 0: #if there is only one moving connection - what happens?
                c_arr_stop = moving_connections[i-1].arr_stop
                c_arr_time = moving_connections[i-1].arr_time_sec
                pi = J[:J.index(moving_connections[i-1])+1]
                
            pi_moving = get_connections(pi)

            # stations visited by the original path
            S_pi = list(set(chain(*[(c.dep_stop, c.arr_stop) for c in pi_moving])))

            # to avoid recomputing path already present in the output. Computes connections to exclude from new CSA run
            C_dev = []
            for Jp in Output:  
                Jp_moving = get_connections(Jp)
                if len(Jp_moving) != 0:
                    if Jp_moving[:i] == pi_moving:
                        C_dev.append(Jp_moving[i])
            
            # updates timetable by removing stations and connections to avoid
            update_timetable(t, S_pi, C_dev)

            # handles the case in which first connection in J is a walk
            if (J[j].trip_id == 'W') & (i == j):
                pi = J[:j+1]
                c_arr_time = J[j].arr_time_sec
                c_arr_stop = J[j].arr_stop
            
            if len(pi)==0:
                c_arr_time = departure_time
                c_arr_stop = departure_station

            # computes deviation path based on the new timetable, new departure station and new departure time
            Q = CSA(stations, t, c_arr_stop, c_arr_time, arrival_station, arrival_time, day)

            # appends deviation path to segment of the orginal path that has been kept
            J_new = pi + Q
            if len(Q) != 0:
                Candidates.append((J_new,i))
        
    return Output


# %% [markdown]
# ## Uncertainty computing

# %% [markdown]
# Read the distribution of the delays

# %%
stats = pd.read_csv('../data/prob_istdaten.csv')

# %% [markdown]
# Read the distribution of the delays

# %%
dict = {'0-20': '20',
        '20-40': '40',
        '40-60': '60',
        '60-80': '80',
        '80-100': '100',
        '100-120': '120',
        '120-140': '140',
        '140-160': '160',
        '160-180': '180',
        '180-200': '200'}
 
stats.rename(columns=dict,
          inplace=True)


# %% [markdown]
# Function to get the number of hours from a number of seconds

# %%
def get_hours_from_second(nbr_of_seconds):
    """nbr_of_seconds is an integer representing a duration in seconds since midnight
    return the corresponding hour as an integer"""
    return int(nbr_of_seconds / 3600)


# %% [markdown]
# Function to calculate the confidence of a trip

# %%
def probability_trip(connections, ist_daten_probs, day):
    
    transport_dict= {'B':'bus','T':'tram','IC':'zug','S':'zug','IR':'zug','PB':'Aerial Tramway','SN':'zug','FAE':'Ferry-boat',
                   'EC':'zug','FUN':'Funicular','RE':'zug', 'BN':'bus','BAT':'Boat','TGV':'zug','EXT':'zug',
                   'ICE':'zug','NJ':'zug','RJX':'zug','KB':'bus', 'W':'Walk'}
    
    # Creation of a dictionary to rename the days
    dict_day = {'0': 'Mon',
        '1': 'Tue',
        '2': 'Wed',
        '3': 'Thu',
        '4': 'Fri',
        '5': 'Sat',
        '6': 'Sun'}
    
    proba_list = []
    
    # Rename the day
    if day in dict_day:
        day = dict_day[day]
    
    #Loop over the connections
    for i in range(len(connections)-1):        
        trip_id_1 = connections[i].trip_id
        connection_2 = connections[i+1]
        trip_id_2 = connection_2.trip_id
        
        # If the trip_id is different, this means we change transportation
        if trip_id_1 != trip_id_2:
            
            # If the trip_id is 'W', this means we walk, special case
            if trip_id_2 == 'W':
                if i+2 == len(connections):
                    
                    spare_time = 120
                    connection_1 = connections[i] #modified
                
                else:
                    # Find the next connection after the walk
                    connection_3 = connections[i+2]
                    connection_1 = connections[i]
                    spare_time = connection_3.dep_time_sec - connection_2.arr_time_sec # spare time with the next connection considering the walk
                
                type_transport_1 = connection_1.route_desc
                
                # Rename the type of transport
                if type_transport_1 in transport_dict:
                    type_transport_1 = transport_dict[type_transport_1]
                    
                # Get the departure hour
                hour_departure = get_hours_from_second(connection_1.dep_time_sec)
                
                # Finding the corresponding delay probability
                for row in ist_daten_probs.iterrows():            
                    hours_list = re.findall(r'\d+', row[1]['index']) #Get the hours
                    
                    
                    # If the type of transport, the day and the hour correspond
                    if (row[1]['index'].find(type_transport_1) != -1) and (row[1]['index'].find(day) != -1) and (hour_departure >= int(hours_list[0])) and (hour_departure < int(hours_list[1])):                
                        row = row[1].drop(labels=['Unnamed: 0', 'index'])
                        indx_1 = 0    
                        
                                            
                        # Loop over the columns to find corresponding delay probability
                        for indx_2, values in row.items():
                            if (spare_time >= int(indx_1)) and (spare_time < int(indx_2)):
                               
                                proba = values
                                break
                            indx_1 = indx_2
                            
                        break
                    else:
                        proba=1 #If no corresponding delay probability, we put 1, this means we don't have any corresponding delay distribution
                        
            else:
                # If the trip_id is not 'W', this means we take a transportation
                connection_1 = connections[i]
                trip_id_1 = connection_1.trip_id
                spare_time = connection_2.dep_time_sec - connection_1.arr_time_sec - 120 #Spare time between the two connections, we substract 120 because csa considers a spare time of 2 minutes to change transportation
                type_transport_1 = connection_1.route_desc
                
                # Rename the type of transport
                if type_transport_1 in transport_dict:
                    type_transport_1 = transport_dict[type_transport_1]
                    
                # Get the departure hour   
                hour_departure = get_hours_from_second(connection_1.dep_time_sec)
               
                # Finding the corresponding delay probability
                for row in ist_daten_probs.iterrows():            
                    # Get the hours
                    hours_list = re.findall(r'\d+', row[1]['index'])
                    
                    # If the type of transport, the day and the hour correspond
                    if (row[1]['index'].find(type_transport_1) != -1) and (row[1]['index'].find(day) != -1) and (hour_departure >= int(hours_list[0])) and (hour_departure < int(hours_list[1])):               
                        row = row[1].drop(labels=['Unnamed: 0', 'index'])
                        indx_1 = 0
                        
                        # Loop over the columns to find corresponding delay probability
                        for indx_2, values in row.items():
                            if (spare_time >= int(indx_1)) and (spare_time < int(indx_2)):
                                
                                proba = values
                                break
                            indx_1 = indx_2
                        break
                    else:
                        proba=1  #If no corresponding delay probability, we put 1, this means we don't have any corresponding delay distribution
        
        
            proba_list.append(proba) #Append the probability to the list
        
    return np.prod(proba_list)


# %% [markdown]
# Function to filter out journeys that do not meet required arrival time and have two walking connection in a row

# %%
def filter_journeys(journeys, arr_time):
    
    filtered_journeys = []
    arr_times = []
    
    for journey in journeys:
#         # to avoid two walking connection in row
#         for i in range(len(journey)-1):        
#             trip_id_1 = journey[i].trip_id
#             trip_id_2 = journey[i+1].trip_id
            
#             if (trip_id_1== trip_id_2) and (trip_id_1 == 'W'):
#                 error = 1
#                 break
        if journey[-1].arr_time_sec <= arr_time:
            # arr_times.append(arr_time -journey[-1].arr_time_sec)
            filtered_journeys.append(journey)
    
    # index_arr_times_sorted = np.argsort(arr_times)
    # index_arr_times_sorted_2 = index_arr_times_sorted[::-1]

    filtered_journeys.sort(key=lambda x: x[-1].arr_time_sec)
    return filtered_journeys


# %% [markdown]
# ## Load stops

# %%
## This file was originally created with stop located within a 30km radius of Zurich HB.
# In the user interface, we only use the ones that are located within 15km of the main station
df_zurich=pd.read_csv('../data/df_zurich.csv')
df_zurich.columns = ['unknown', 'stop_id', 'stop_name', 'latitude', 'longitude', 'location_type', 'parent_station', 'center_dist']
df_zurich.drop(columns=['location_type','parent_station'],inplace=True)


# %%
def unique_sorted_values(array):
    unique = array.unique().tolist()
    unique.sort()
    return unique


# %%
def seconds_to_h_m(time):
    '''Convert seconds time to H:M'''
    hours, remainder = divmod(time, 3600)  # Convert to hours
    minutes, _ = divmod(remainder, 60)  # Convert remainder to minutes
    
    return hours, minutes


# %%
# Relation between the route_desc of timetable and the real transport name
ttype_dict= {'B':'Bus','T':'Tram','IC':'Train','S':'Train','IR':'Train','PB':'Aerial Tramway','SN':'Train','FAE':'Ferry-boat',
                   'EC':'Train','FUN':'Funicular','RE':'Train', 'BN':'Bus','BAT':'Boat','TGV':'Train','EXT':'Train',
                   'ICE':'Train','NJ':'Train','RJX':'Train','KB':'Bus', 'W':'Walk'}

# %%
# Color dictionary which associates a particular color for the most common ttype
color_dict = {
  "Bus": "#636EFA",
  "Tram": "#EF553B",
  "Train": "#00CC96",
  "Other":"#808080",
"Walk":"#FFD700",
"Funicular":"#808080",
"Boat":"#808080",
"Ferry-boat":"#808080",
"Aerial Tramway":"#808080",
}

# %% [markdown]
# ## Widget setup
# ### Generation of the map

# %%
import plotly.graph_objects as go
# Function definition for plotting the travel
def plot_map(journey_df):
    fig = go.Figure()
    
    ttype_list = []
    for i in range(len(journey_df)):
        # Below for segment between stations
        
        if journey_df.iloc[i]['ttype'] not in ttype_list:
            ttype_list.append(journey_df.iloc[i]['ttype'])
            display_legend=True
        else:
            display_legend=False
        
        fig.add_trace(go.Scattermapbox(
            #mode = "markers",
            mode= "lines",
            name= journey_df.iloc[i]['ttype'],
            lat = [ journey_df.iloc[i]['dep_latitude'],journey_df.iloc[i]['arr_latitude'] ],#journey_df.iloc[i:i+2]['latitude'].tolist(),
            lon = [ journey_df.iloc[i]['dep_longitude'],journey_df.iloc[i]['arr_longitude'] ],#journey_df.iloc[i:i+2]['longitude'].tolist(),
            marker = {'color': color_dict[journey_df.iloc[i]['ttype']], #Here put the color associated with the type of transport!
                      "size": 10},
            showlegend=display_legend,
            hoverinfo='none'
        ))
    
    
    #Generate the dataframe of stop where you actually have to change
    change_stops = journey_df.groupby('trip_id').first()
    
    #Need to append last station
    transit_lat=change_stops['dep_latitude'].tolist()
    transit_lat.append(journey_df.iloc[-1]['arr_latitude'])
    transit_lon=change_stops['dep_longitude'].tolist()
    transit_lon.append(journey_df.iloc[-1]['arr_longitude'])
    transit_stop_name=change_stops['dep_stop_name'].tolist()
    transit_stop_name.append(journey_df.iloc[-1]['arr_stop_name'])
    
    fig.add_trace(go.Scattermapbox(
            mode = "markers",
            lat = transit_lat,
            lon = transit_lon,
            hovertext = transit_stop_name,
            hovertemplate="<br>".join([ "Stop: %{hovertext}","<extra></extra>"]),
            marker = {'color': 'black', 
                      "size": 10},
            showlegend=False
        ))   
  
   
    # To display the maps at the good location
    min_lat=journey_df['dep_latitude'].min()
    max_lat=journey_df['dep_latitude'].max()
    middle_lat=min_lat+(max_lat-min_lat)/2
    range_lat = max_lat-min_lat
    
    min_lon=journey_df['dep_longitude'].min()
    max_lon=journey_df['dep_longitude'].max()
    middle_lon=min_lon+(max_lat-min_lat)/2
    range_lon = max_lon-min_lon
    
    max_range= max(range_lat,range_lon)
    
    if max_range<0.04:
        zoom=11
    elif max_range<0.05:
        zoom=10.75
    elif max_range<0.07:
        zoom=10.5
    elif max_range<0.08:
        zoom=10.25
    elif max_range<0.09:
        zoom=10
    else:
        zoom=9
    
    fig.update_layout(margin ={'l':0,'t':0,'b':0,'r':0},
                      mapbox = {
                          'center': {'lat': middle_lat,'lon': middle_lon},
                          'style': "stamen-terrain",
                          'zoom': zoom},
                      width=750,
                      height=300,)
    fig.show()

# %% [markdown]
# ### Generation of the user interface

# %%
from ipywidgets import HBox, VBox, Layout,Button
import ipywidgets as widgets
import numpy as np
from IPython.display import display, clear_output
import datetime
import math

class color:
   RED = '\033[91m'
   BOLD = '\033[1m'
   END = '\033[0m'

output = widgets.Output()

#Start destination
#Select stops inside 15km radius for start and stop options
start_w = widgets.Dropdown(
    options=unique_sorted_values(df_zurich[df_zurich['center_dist']<=15]['stop_name']),
    value='Zürich, Kronenstrasse',
    description='From:')
#Stop destination
stop_w = widgets.Dropdown(
    options=unique_sorted_values(df_zurich[df_zurich['center_dist']<=15]['stop_name']),
    value='Zürich, Chinagarten',
    description='To:')
#Date Picker
date_w = widgets.DatePicker(description='Pick a Date',value=datetime.date.today(),)
#Hour selector
dep_hour_w = widgets.Dropdown(
    options=np.arange(24),
    value=14,
    description='Departure H:',
    layout=Layout(width='140px'))
#Minute selector
dep_min_w = widgets.Dropdown(
    options=np.arange(60),
    value=0,
    description='M:',
    layout=Layout(width='140px'))

#Hour selector
arr_hour_w = widgets.Dropdown(
    options=np.arange(24),
    value=16,
    description='Arrival H:',
    layout=Layout(width='140px'))
#Minute selector
arr_min_w = widgets.Dropdown(
    options=np.arange(60),
    value=0,
    description='M:',
    layout=Layout(width='140px'))

#slider for uncertainty
confidence_w = widgets.IntSlider(
    value=80,
    min=0,
    max=100,
    step=1,
    description='Min prob.:',
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
time_items_dep=[dep_hour_w, dep_min_w]
time_items_arr=[arr_hour_w, arr_min_w]

sub_box_dep = HBox(children=time_items_dep)
sub_box_arr = HBox(children=time_items_arr)
#Overall widget box
items = [start_w,stop_w,date_w,sub_box_dep,sub_box_arr, confidence_w]
big_box = VBox(children=items) # Vertical display

display(big_box) 
display(enter_b, output)

def on_button_clicked(b):
    with output:
        clear_output()
        
        
        departure_stop= df_zurich.loc[df_zurich['stop_name'] == start_w.value, 'stop_id'].values[0]
        final_stop=df_zurich.loc[df_zurich['stop_name'] == stop_w.value, 'stop_id'].values[0]
        
        # Get the day of the week
        day_of_week = date_w.value.weekday()
        
        des_departure=dep_hour_w.value * 3600 + dep_min_w.value * 60
        des_arrival=arr_hour_w.value * 3600 + arr_min_w.value * 60
        
        if des_arrival<des_departure:
            print('Please provide an arrival time that is after the departure time.')
            
        else:
            print('Please wait while we process your query',end='\r')

     
            reset_timetable(ls_timetable, original_length)
            
            # Compute K journeys with CSA
            journeys_total= CSA_multiple(stations, ls_timetable,departure_stop,des_departure,final_stop,des_arrival,day_of_week, k=10)

            if not journeys_total: # If no trip were found
                print(color.BOLD+'Unforturnately no connections was found for your desired trip between {} and {}.\nPlease try another journey.'.format(start_w.value,stop_w.value)+color.END)
            
            # Filter the trips that arrive too lat
            journeys=filter_journeys(journeys_total,des_arrival)
        
            
            if not journeys: # If no trip were found
                print(color.BOLD+'Unforturnately no connections was found for your desired trip between {} and {}.\nPlease try another journey.'.format(start_w.value,stop_w.value)+color.END)
            else:
                valid_options=0
                for option_n in range(len(journeys)):
                    
                    #Compute the probability for a given trip
                    prob_of_trip= probability_trip(journeys[option_n], stats, str(day_of_week))
                    
                    # If the trip has a lower probability than asked by user, don't show it
                    if prob_of_trip<((confidence_w.value)/100):  
                        continue
                    valid_options+=1
                    connections_df=pd.DataFrame(journeys[option_n])
                    tot_trav_time=(connections_df.arr_time_sec.iloc[-1]-connections_df.dep_time_sec.iloc[0])
                    connections_df['ttype']=connections_df.route_desc.replace(ttype_dict)

                    #Merging with the zurich stops info
                    transit_df=pd.merge(connections_df, df_zurich.add_prefix('dep_'), left_on='dep_stop',right_on='dep_stop_id', how='left').drop(columns=['dep_stop_id'])
                    transit_df=pd.merge(transit_df, df_zurich.add_prefix('arr_'), left_on='arr_stop',right_on='arr_stop_id', how='left').drop(columns=['arr_stop_id'])

                    # Update the trip id 'W' with W_1, W_2,... so that it is differentiable for groupby later on
                    # Get the indices of the rows where the column value is 'W'
                    indices = transit_df.index[transit_df['trip_id'] == 'W']

                    # Iterate over the indices and update the column values
                    counter = 1
                    for index in indices:
                        transit_df.at[index, 'trip_id'] = f"W_{counter}"
                        counter += 1

                    change_stops_dep = transit_df.groupby('trip_id').first().sort_values(by=['dep_time_sec']).reset_index()
                    change_stops_arr = transit_df.groupby('trip_id').last().sort_values(by=['arr_time_sec']).reset_index()
                    
                   

                    ##FIRST STOP
                    departure_H, departure_M = seconds_to_h_m(change_stops_dep.iloc[0]['dep_time_sec'])
                    if departure_M <= 9:
                        departure_M = '0'+str(departure_M)
                    arrival_H, arrival_M = seconds_to_h_m(change_stops_arr.iloc[0]['arr_time_sec'])
                    if arrival_M <= 9:
                        arrival_M = '0'+str(arrival_M)
                    final_arrival_H, final_arrival_M = seconds_to_h_m(change_stops_arr.iloc[-1]['arr_time_sec'])
                    if final_arrival_M <= 9:
                        final_arrival_M = '0'+str(final_arrival_M)
                    print(color.BOLD+'Option {} going from {} to {}:'.format(valid_options,change_stops_dep.iloc[0]['dep_stop_name'],change_stops_arr.iloc[-1]['arr_stop_name'])+color.END) 
                    print(color.BOLD+'Travel time: {} min. Departure at: {}:{}. Arrival at {}:{}'.format(int(tot_trav_time/60),departure_H,departure_M,final_arrival_H,final_arrival_M))
                    print('Probability of a successful trip: '+color.RED+'{:.1f}%\n'.format(prob_of_trip*100)+color.END)

                    if change_stops_dep.iloc[0]['ttype'] != 'Walk':
                        print('•Leave your departure stop ({}) at {}:{} \n taking the {} {} reaching {} at {}:{}.\n' \
                            .format(change_stops_dep.iloc[0]['dep_stop_name'],departure_H,departure_M, change_stops_dep.iloc[0]['ttype'].lower(), change_stops_dep.iloc[0]['trip_id'], change_stops_arr.iloc[0]['arr_stop_name'],arrival_H,arrival_M))
                    else:
                        trav_time= math.floor((change_stops_dep.iloc[0]['arr_time_sec']-change_stops_dep.iloc[0]['dep_time_sec'])/60)
                        print('•Walk {} min in order to reach {}.\n'.format(trav_time,change_stops_dep.iloc[0]['arr_stop_name']))

                    ## TRANSIT STOPS
                    for i in range(1,len(change_stops_dep)-1):
                        if change_stops_dep.iloc[i]['ttype'] != 'Walk':
                            departure_H, departure_M = seconds_to_h_m(change_stops_dep.iloc[i]['dep_time_sec'])
                            if departure_M <= 9:
                                departure_M = '0'+str(departure_M)
                            arrival_H, arrival_M = seconds_to_h_m(change_stops_arr.iloc[i]['arr_time_sec'])
                            if arrival_M <= 9:
                                arrival_M = '0'+str(arrival_M)
                            print('•At {}, change for the {} {}\n departing at {}:{} and arriving in {} at {}:{}.\n'.format(change_stops_dep.iloc[i]['dep_stop_name'],change_stops_dep.iloc[i]['ttype'].lower(), change_stops_dep.iloc[i]['trip_id'],departure_H,departure_M, change_stops_dep.iloc[i+1]['dep_stop_name'],arrival_H,arrival_M))

                        else:
                            trav_time= math.floor((change_stops_dep.iloc[i]['arr_time_sec']-change_stops_dep.iloc[i]['dep_time_sec'])/60)
                            print('•Walk from {} to {} (~{} min walk).\n'.format(change_stops_dep.iloc[i]['dep_stop_name'],change_stops_dep.iloc[i]['arr_stop_name'],trav_time))

                    ## FINAL STOP
                    if len(change_stops_dep)>1:
                        if change_stops_dep.iloc[-1]['ttype'] != 'Walk':
                            departure_H, departure_M = seconds_to_h_m(change_stops_dep.iloc[-1]['dep_time_sec'])
                            if departure_M <= 9:
                                departure_M = '0'+str(departure_M)
                            arrival_H, arrival_M = seconds_to_h_m(change_stops_arr.iloc[-1]['arr_time_sec'])
                            if arrival_M <= 9:
                                arrival_M = '0'+str(arrival_M)
                            print('•At {}, change for the {} {}\n departing at {}:{} toward {}.'.format(change_stops_dep.iloc[-1]['dep_stop_name'],change_stops_dep.iloc[-1]['ttype'].lower(), change_stops_dep.iloc[-1]['trip_id'],departure_H,departure_M, change_stops_dep.iloc[-1]['arr_stop_name']))
                            print(' You will arrive at your desired final destination at {}:{}.\n'.format(arrival_H,arrival_M))
                        else:
                            trav_time= math.floor(change_stops_dep.iloc[-1]['arr_time_sec']-change_stops_dep.iloc[-1]['dep_time_sec'])/60
                            print('•Walk {} min to reach your final destination ({}).\n'.format(int(trav_time),change_stops_dep.iloc[-1]['arr_stop_name']))
                    plot_map(transit_df)
                    
                    # Limit to a certain number of offered trip (have noticed some displaying bug with the widget otherwise)
                    if valid_options>5:
                        break
                
        

enter_b.on_click(on_button_clicked)

# %%
