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
# # Constructing the timetable dataframe

# %% [markdown]
# We are going to construct the timetable dataframe based on the 3rd May 2023.

# %% [markdown]
# ## Setup

# %%
import os
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", 50)
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
import plotly.graph_objects as go
warnings.simplefilter(action='ignore', category=UserWarning)

# %%
# %load_ext sparkmagic.magics

# %%
import os
from IPython import get_ipython
username = os.environ['RENKU_USERNAME']
server = "http://iccluster044.iccluster.epfl.ch:8998"

# set the application name as "<your_gaspar_id>-homework3"
get_ipython().run_cell_magic(
    'spark',
    line='config', 
    cell="""{{ "name": "{0}-final-project3", "executorMemory": "4G", "executorCores": 4, "numExecutors": 10, "driverMemory": "4G" }}""".format(username)
)

# %%
get_ipython().run_line_magic(
    "spark", f"""add -s {username}-final-project3 -l python -u {server} -k"""
)

# %% language="spark"
# import pyspark.sql.functions as F
# from pyspark.sql import Window
# from pyspark.sql.functions import col, expr

# %% [markdown]
# ## Loading the stops in a cercle of 18km around Zurich BH

# %% language="spark"
# #Loading the file previously preprocessed
# zurich_18km = spark.read.csv('/user/nmuenger/work/zurich_18km_true.csv', header=True, encoding='utf8')

# %% language="spark"
# #Renaming the columns
# ls = ['unknown', 'stop_id', 'stop_name']
# zurich_18km = zurich_18km.toDF(*ls) 

# %% [markdown]
# ## Merging the dataframes

# %% [markdown]
# ### Starting by loading necessary files

# %% language="spark"
# #Loading the trips.txt file
# trips = spark.read.csv("/data/sbb/part_csv/timetables/trips/year=2023/month=05/day=03/trips.txt", header=True, encoding='utf8')

# %% language="spark"
# #Loading the stop_times.txt file
# stop_times = spark.read.csv("/data/sbb/part_csv/timetables/stop_times/year=2023/month=05/day=03/stop_times.txt", header=True, encoding='utf8')

# %% language="spark"
# #Loading the stop.txt file
# stops = spark.read.csv("/data/sbb/part_csv/timetables/stops/year=2023/month=05/day=03/stops.txt", header=True, encoding='utf8')

# %% language="spark"
# #Loading the routes.txt file
# routes = spark.read.csv("/data/sbb/part_csv/timetables/routes/year=2023/month=05/day=03/routes.txt", header=True, encoding='utf8')

# %% [markdown]
# ### Merging the files

# %% language="spark"
# #Merging the trips and stop_times dataframe based on the trip_id
# temp=stop_times.join(trips, on='trip_id',how='inner')
#
# #Dropping unnecessary columns
# first_merge=temp.drop('direction_id','block_id')

# %% language="spark"
# #Merging the first_merge and stops dataframe based on the stop_id
# second_merge=first_merge.join(stops,on='stop_id',how='inner')

# %% [markdown]
# Before finalizing the merge, we determine the stop sequences for each trip and the time of departure and arrival at each stop.

# %% language="spark"
# #Determining the stops sequence
# windowSpec = Window.partitionBy("trip_id").orderBy("stop_sequence")
# second_merge = second_merge.withColumn("departure_stop", F.lag(second_merge["stop_id"],1).over(windowSpec))
# second_merge = second_merge.withColumn("arrival_stop", second_merge["stop_id"])

# %% language="spark"
# #Computing the departure and arrival time
# windowSpec = Window.partitionBy("trip_id").orderBy("stop_sequence")
# second_merge = second_merge.withColumn("departure_time_bis", F.lag(second_merge["departure_time"],1).over(windowSpec))
# second_merge = second_merge.withColumn("arrival_time_bis", second_merge["arrival_time"])

# %% language="spark"
# #Assuring that the time value are valid
# second_merge = second_merge.filter(col('departure_time_bis').isNotNull())

# %% [markdown]
# Now, the third merge can be computed.

# %% language="spark"
# #Merging the second merge and routes dataframe with the route_id
# third_merge=second_merge.join(routes[['route_id','route_desc']],on='route_id',how='inner')

# %% language="spark"
# #Loading the calendar.txt
# calendar = spark.read.csv("/data/sbb/part_csv/timetables/calendar/year=2023/month=05/day=03/calendar.txt", header=True, encoding='utf8')

# %% language="spark"
# #Merging the third merge and calendar dataframe with the service_id
# fourth_merge = third_merge.join(calendar, on='service_id', how="inner")

# %% [markdown]
# ## Filtering the merge dataframe with the chosen stops

# %% [markdown]
# Let's filter the useful columns of the fourth merge dataframe

# %% language="spark"
# timetable=fourth_merge.select('stop_id', 'trip_id', 'stop_name',   'departure_stop', 'arrival_stop', 'departure_time_bis', 'arrival_time_bis', 'route_desc', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday')                        

# %% language="spark"
# # selecting the stops of interest
# zurich_stops = set(zurich_18km.rdd.map(lambda x: x.stop_id).collect())
# zurich_timetable=timetable.filter(F.col('stop_id').isin(zurich_stops))

# %% language="spark"
# #saving the Zurich stops timetable
# zurich_timetable.write.csv('/user/nmuenger/work/timetable_zh.csv',header=True)

# %%
# %spark cleanup


# %%
