#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 15:00:43 2019

@author: twillia2
"""
import dask.array as da
import datetime as dt
import geopandas as gpd
import h5py
import numpy as np
import os
import pandas as pd
from pygds.utilities import make_pg_connection
from shapely.geometry import Point
from tqdm import tqdm
from dask.distributed import Client


def append_zip_tz(crddf):
    """Take a data frame with coordinates, append US zip codes and time zones
    and return a new data frame.
    """
        
    # Connect to my new postgres databases!
    con, cur, engine = make_pg_connection(user="twillia2",
                                          host='1lv11gispg02.nrel.gov')
    
    # US Zip Code Tabulation Areas in 2017
    print("Reading zip code tabulation area data frame from postgres...")
    query = """select geoid10, the_geom_4326 from twillia2.us_zcta_2017;"""
    zipdf = gpd.read_postgis(query, engine, geom_col="the_geom_4326")
    zipdf.columns = ["zip", "geometry"]

    # Filter data geoframe for places within zip codes...
    print("Merging zip codes with resource data coordinates...")
    crddf.columns = ["lat", "lon"]
    crddf["geometry"] = crddf.apply(to_point, axis=1)
    crs={"init": "epsg:4326"}
    crddf = gpd.GeoDataFrame(crddf, geometry="geometry", crs=crs)

    # Using points misses quite a few counties, let's use a buffer
    buffer = crddf.buffer(.02)
    buffer = gpd.GeoDataFrame(buffer)
    buffer["gid"] = buffer.index
    buffer.columns = ["geometry", "gid"]
    buffer.crs = crs

    # Some zip codes will have multiple resource points, taking the first here
    metadf = gpd.sjoin(zipdf, buffer)
    metadfo = gpd.sjoin(zipdf, crddf)
    zipids = metadf[["zip", "index_right"]]
    zipids.columns = ["zip", "gid"]
    zipids = zipids.reset_index(drop=True)
    group = zipids.groupby("zip")
    zipids["gid"] = group["gid"].transform(lambda x: x.iloc[0])  # take the first?
    zipids = zipids[["gid", "zip"]].drop_duplicates()

    # Join the resource coordinates back to this data frame
    metadf2 = pd.merge(zipids, crddf)
    metadf2 = gpd.GeoDataFrame(metadf2, geometry="geometry")

    # We need to adjust timezones to reflect local times, Working on this
    print("Attaching local time zone adjustments relative to UTC...")
    metadf = time_zones(metadf)
    metadf = metadf.drop("index_right", axis=1)
    metadf = metadf.sort_values(by="resource_index", ascending=True)

    # Add ztca centroids ?
    zipdf["centroid"] = zipdf['geometry'].centroid
    centroid_dict = dict(zip(zipdf['zip'], zipdf['centroid'])) 
    metadf['centroid'] = metadf['zip'].map(centroid_dict)

    return metadf



def day_stats(meta, resource, time_index, hour_range=(7, 17)):
    """Apply time zone adjustments, filter values with the hour range, and
    determine the minimum and maximum values of the resource array at each
    point.
    
    Paramaters
    ----------
    meta (pandas.core.frame.DataFrame)
    resource (numpy.ndarray)
    time_index (list)
    
    Returns
    -------
   
    
    """

    # Here are the times in UTC
    print("Converting date strings to datetime objects...")
    time_array = np.array([t.astype("str")[:-13] for t in time_index])
    time = [dt.datetime.strptime(t, "%Y-%m-%dT%H:%M") for t in time_array]

    # Add time to end of temp - memory management is important here.
    print("Setting up local time filter process...") 
    adj = np.array([meta["zone"].values])
    combined = da.concatenate([resource, adj]).compute()

    # This needs to be on disk to avoid memory spikes
    with h5py.File("/scratch/twillia2/fleet/temp.h5", "w") as temp:
        temp.create_dataset(name="temperature_tz", data=combined)
    del combined

    # Use Dask distributed to schedule workers
    with Client() as client:
        # Print IP address of dask dashboard?
        print("Calculating summary statistics. \nDask Scheduler " + 
              str(client).replace("<", "").replace(">", "") + 
              "\nDask Diagnostics Dashboard: 'http://127.0.0.1:8787/status'") 
        
        with h5py.File("/scratch/twillia2/fleet/temp.h5", "r") as ds:
            combined = da.from_array(ds["temperature_tz"], chunks="auto")
            
            filtered = np.apply_along_axis(tz_filter, axis=0, arr=combined,
                                           time_list=time, time_range=(7, 17))
            min_values = filtered.min(axis=0).compute()
            max_values = filtered.max(axis=0).compute()
            mean_values = filtered.mean(axis=0).compute()

    meta["max_value"] = max_values
    meta["min_value"] = min_values
    meta["mean_values"] = mean_values

    return meta


def to_point(row):
    """Create a point object from a row with 'lat' and 'lon' columns"""
    point = Point((row["lon"], row["lat"]))
    return point


def when(lst, time_of_day):
    """Find when the maximum value occurred at each point"""
    # I combined the temperature and max temperature arrays
    value = lst[-1]
    lst = lst[:-1]

    # Find all index values where the max was reached
    indices = np.where(lst == value)[0]

    # Find the time of day these were met
    times = list(time_of_day[indices])

    # Find the modal time of day
    mode = max(set(times), key=times.count)

    # Done.
    return mode


def local_times(coords, time_index, hour_range=(7, 17)):
    """Return a 2d array of local date times for a set of points between
    8 AM and 5 PM.
    """
    # Day One
    t1 = time_index[0][:16].decode("utf-8")
    d1 = dt.datetime.strptime(t1, "%Y-%m-%dT%H:%M") 
    coords["day1"] = coords["zone"].apply(lambda x: d1 + dt.timedelta(hours=x))

    # An array with every day in the hour range would be useful...
    # An array of first days
#    first_days = [pd.Timestamp(d).to_pydatetime() for d in coords["day1"]]
    first_days = coords["day1"].values

    # Every half increment in a day for one start date
    rt = range(len(time_index))
    periods = [30 * i for i in rt]

    def single(d):
        return np.array([d + np.timedelta64(periods[i], "m") for i in rt])

    # Longest way
    ndarray = []
    for i in tqdm(range(len(first_days))):
        array = single(first_days[i])
        ndarray.append(array)

    # With dask?
    ddays = da.from_array(first_days, chunks="auto")
    fddays = da.apply_along_axis(single, 0, ddays, dtype="<M8[ns]")


def time_zones(df):
    """Append time zone information to data frame with lat lon"""

    # Get time zone file - only so many remotely accessible like this
    if not os.path.exists("data/time_zones.shp"):
        timezones = gpd.read_file("https://www.naturalearthdata.com/" +
                                  "http//www.naturalearthdata.com/download/" +
                                  "10m/cultural/ne_10m_time_zones.zip")
        timezones.to_file("data/time_zones.shp")
    else:
        timezones = gpd.read_file("data/time_zones.shp")

    # Filter for needed columns
    timezones = timezones[["zone", "geometry"]]

    # Make sure that data frame (df) is a geodataframe
    if type(df) != gpd.geodataframe.GeoDataFrame:
        df["geometry"] = df.apply(to_point, axis=1)
        df = gpd.GeoDataFrame(df, geometry="geometry",
                              crs={'init': 'epsg:4326'})

    # Join these together
    df = gpd.sjoin(df, timezones, op="within", how="left")

    return df

def tz_filter(value_list, time_list, time_range):
    """This list needs to have a time zone adjustment appended to the last
    position. This is what it needs to do:

        1) Separate the values from the time zone adjustment
        2) adjust the time list using this adjustment
        3) Filter values within the time range
        4) return

    sample arguments:
        value_list = temp_special[:, 0]
        time_list = time
        time_range = (7, 17)
    """
    # Separate the values from the adjustment, sorry if that's confusing
    adj = value_list[-1]
    values = value_list[:-1]

    # Adjust times (this value needs to be a local UTC to local diff in hours)
    times = np.array([t + dt.timedelta(hours=adj) for t in time_list])
    hours = np.array([t.hour for t in times])
    time1 = time_range[0]
    time2 = time_range[1]
    day_indices = np.where((hours >= time1) & (hours <  time2))[0]

    # There are 2 points that returned nothing
    if len(values) > 0:   
        values = values[day_indices]
    else:
        values = np.array([-9999 for i in day_indices])

    # Should we try to return the times as well? Should we do everything here?
    # day_times = times[day_indices]
    # times = [dt.datetime.strftime(t, "%Y-%m-%dT%H:%M") for t in day_times]

    # Would this create two arrays?
    return values  #, times