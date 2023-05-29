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
# # Distances between stops calculation
#
# In this notebook we want to calculate the distances between the stops in zurich (df_zurich). <br>
# We will keep only the stops that are closer than 500 meters between each other since this is the total distance which is allowed in our trips and that our algorithm should be considering.

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
    cell="""{{ "name": "{0}-final-project6", "executorMemory": "4G", "executorCores": 4, "numExecutors": 10, "driverMemory": "4G" }}""".format(username)
)

# %%
get_ipython().run_line_magic(
    "spark", f"""add -s {username}-final-project6 -l python -u {server} -k"""
)

# %% [markdown]
# # Adding distances
#
# We download the data from the stops in zurich. We keep the data regarding the stop id, it's longitude and latitude.

# %% language="spark"
# from pyspark.sql.functions import col
#
# df_zurich = spark.read.csv('/user/romanato/work/df_zurich.csv', header = True, sep = ',', inferSchema = True)
# ls = ['unknown', 'stop_id', 'stop_name', 'latitude', 'longitude', 'location_type', 'parent_station', 'center_dist']
# df_zurich = df_zurich.toDF(*ls) #renaming the columns
# df_zurich = df_zurich.filter(df_zurich['center_dist'] <= 18)
# df_zurich.show(5)

# %% language="spark"
# #-o df -n -1
# df = df_zurich.select(col('latitude'), col('longitude'), col('stop_id'))

# %%
#df.to_csv('stops.csv')

# %% [markdown]
# We take all the combination between the stops.

# %% language="spark"
# # we combine the data of all the stops together in order to have all the combination inn between stops.
# stops_data1 = df.withColumnRenamed('stop_id', 'stop_id1').withColumnRenamed('latitude', 'latitude1').withColumnRenamed('longitude', 'longitude1')
# stops_data2 = df.withColumnRenamed('stop_id', 'stop_id2').withColumnRenamed('latitude', 'latitude2').withColumnRenamed('longitude', 'longitude2')
# combined_df = stops_data1.crossJoin(stops_data2)
#
# # remove the combinnation of the same stop.
# combined_df = combined_df.filter(combined_df.stop_id1 != combined_df.stop_id2)
# combined_df.show(10)

# %% [markdown]
# Calculation of the distance from degree of the longitude and latitude of the stops to km. In this case we assume the Earth is flat since for the sake of the excercise (Zurich is a limited area) it is a good approximation.

# %% language="spark"
# from pyspark.sql.functions import abs, cos, sin, sqrt, radians
#
# # calculation of longitude and latitude in kilometers.
# df_with_distance = combined_df.withColumn(
#     "dx", abs(combined_df.longitude1 - combined_df.longitude2) * (40075.0 / 360.0)).withColumn(
#     "dy", abs(combined_df.latitude1 - combined_df.latitude2) * (40075.0 / 360.0) * cos(radians(combined_df.latitude2)))
#
# # calculation of the distance using Pitagora theorem.
# df_with_distance = df_with_distance.withColumn(
#     "distance", sqrt(df_with_distance.dx*df_with_distance.dx + df_with_distance.dy *df_with_distance.dy)  # Earth's radius in kilometers
# ).select(col('stop_id1'), col('stop_id2'), col('distance'), col('latitude1'), col('longitude1'), col('latitude2'), col('longitude2'))
#
# df_with_distance.show(10)

# %% language="spark"
#
# #stops_with_lonlat = df_with_distance.sample(0.0005)
# #stops_with_lonlat.count()

# %% [markdown]
# Save a subset for the evaluation function sake.

# %% magic_args="-o stops_with_lonlat -n -1" language="spark"
#
# #stops_with_lonlat

# %%
#stops_with_lonlat.to_csv('stops_with_lonlat.csv')

# %% [markdown]
# Filter on the stops that are closer than 500 meters.

# %% language="spark"
# df_with_distance = df_with_distance.filter(df_with_distance.distance <= 0.5).select(['stop_id1', 'stop_id2', 'distance'])
# df_with_distance.show(5)

# %% [markdown]
# Calculate the transfer time by converting the distance in km to seconds assuming the speed of 50m/minute and assuming that each change requires a 2 minutes additional time. The transfer time is diplayed in seconds.

# %% magic_args="-o df_with_transfer_time -n -1" language="spark"
# from pyspark.sql.functions import expr
#
# # Assuming you have a DataFrame called 'df_with_distance' with a 'distance' column
#
# df_with_transfer_time = df_with_distance.withColumn(
#     "transfer_time", expr("floor(60 * distance / 0.05)")
# )
#
# df_with_transfer_time = df_with_transfer_time.select(col('stop_id1'), col('stop_id2'), col('walking_time'))
# df_with_transfer_time.show()

# %%
# save the dataset
df_with_transfer_time.to_csv('df_with_transfer_time.csv', index = False)
