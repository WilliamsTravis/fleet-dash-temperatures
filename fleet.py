# -*- coding: utf-8 -*-
"""
Create a csv with average minimum and maximum temperatures for all zip codes in
all 50 states. "Average min” and “average max” in this case should be
determined by pulling the temperature time series for each NSRDB location
within the zip code, filtering the time series to between 800-1700 local time
(recommend numpy.roll) and returning a minimum and maximum temp for the zip
Code at one sample point that can acts as the zip code proxy. Only consider
NSRDB pixels if their centroid is within the given zip code. We need to include
coordinates for the sample point. It wouldn’t hurt to throw in local time of
each temp observation.

Note:
    This hasn't been run all the way through and the centroid step isn't even
    built in yet. It will take some hand holding to get it to run.

Notes:
    Adjust for time zones
    Return representative min/max by mode and median
        return zcta centroid coordinates
  
    Return overall median and mode values, too

    Also, include the date with the local times to help trouble shoot
    Why numpy roll?
 
Created on Tue Dec 31 08:23:24 2019

@author: twillia2
"""
import dask.array as da
import geopandas as gpd
import h5py as hp
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pygds.utilities import make_pg_connection
from shapely.geometry import Point

# Make a package?
from functions import append_zip_tz, day_stats

# Expand the Pandas printout range
pd.set_option("display.max_rows", 50)
pd.set_option("display.max_columns", 15)
pd.set_option("display.width", 700)

# We need air_temperature to stay on disk, pull in time_index
ds = hp.File("/scratch/twillia2/fleet/nsrdb_2017.h5", "r")
timezone = ds["time_index"].attrs["time_zone"]
time_index = ds["time_index"][:]
crddf = pd.DataFrame(ds["coordinates"][:])
temp = da.from_array(ds["air_temperature"], chunks="auto")

# Get the zip codes, coordinates, and time zones
con, cur, engine = make_pg_connection(user="twillia2",
                                      host='1lv11gispg02.nrel.gov')

# US Zip Code Tabulation Areas in 2017
query = """select geoid10, the_geom_4326 from twillia2.us_zcta_2017;"""
zipdf = gpd.read_postgis(query, engine, geom_col="the_geom_4326")
zipdf = gpd.GeoDataFrame(zipdf, geometry="geometry")
zipdf["centroid"] = zipdf.geometry.centroid
zipdf.columns = ["zip", "geometry", "centroid"]
con.close()
engine.close()
meta_original = append_zip_tz(crddf, zipdf)

# Now, subset the temperature data by the new resource index
resource_index = meta_original["resource_index"].values
resource = temp[:, resource_index]

# Append summary statistics for values between 8 AM and 5 PM, local time
meta = day_stats(meta=meta_original, resource=resource, time_index=time_index,
                 hour_range=(7, 17))
ds.close()

# Fill in missing zip codes with neighboring values
# meta = pd.read_csv("/scratch/twillia2/fleet/final_long.csv")
meta["zcta"] = meta["zcta"].apply(lambda s: f"{s:05d}")
meta["centroid"] = meta["centroid"].apply(lambda x: Point(x))
caught_zips = meta['zcta'].unique()
all_zips = zipdf['zip'].unique()
missing_zips = list(set(all_zips).difference(caught_zips))
mzipdf = zipdf[zipdf["zip"].isin(missing_zips)]
mzipdf.columns = ["zcta", "geometry", "centroid"]
mzipdf["lat"] = mzipdf["centroid"].apply(lambda c: c.y)
mzipdf["lon"] = mzipdf["centroid"].apply(lambda c: c.x)
mzipdf["utc_zone"] = np.nan

mzipdf = mzipdf[["zcta", "lat", "lon", "centroid", "utc_zone"]]
meta = gpd.GeoDataFrame(meta, geometry="centroid")
mzipdf = gpd.GeoDataFrame(mzipdf, geometry="centroid")
mzipdf = mzipdf.reset_index(drop=True)
n1 = np.array(list(zip(mzipdf.geometry.x, mzipdf.geometry.y)))
n2 = np.array(list(zip(meta.geometry.x, mzipdf.geometry.y)))
btree = cKDTree(n2)
dist, idx = btree.query(n1, k=1)
new_values = pd.DataFrame({'distance': dist.astype(int),
                           'tmin' : meta.loc[idx, "tmin"],
                           'tmax' : meta.loc[idx, "tmax"],
                           'tmean' : meta.loc[idx, "tmean"],
                           'nsrdb_id': meta.loc[idx, "nsrdb_id"]})
new_values = new_values.reset_index(drop=True)
newdf = mzipdf.join(new_values)
newdf = newdf[['zcta', 'nsrdb_id', 'lat', 'lon', 'centroid', 'utc_zone',
               'tmax', 'tmin', 'tmean']]

# Join these
meta = pd.concat([meta, newdf])

# We need to get the utc zone from somewhere else!


# Group by zip, calculate modes, and clean column names for delivery
meta = meta.drop("geometry", axis=1)
group = meta.groupby("zcta")
meta["tmin_mode"] = group["tmin"].transform(lambda x: pd.Series.mode(x)[0])
meta["tmax_mode"] = group["tmax"].transform(lambda x: pd.Series.mode(x)[0])
meta["tmax_mean"] = group["tmax"].transform("mean")
meta["tmin_mean"] = group["tmin"].transform("mean")
meta["tmax_max"] = group["tmax"].transform("max")
meta["tmin_min"] = group["tmin"].transform("min")
meta["total_tmean"] = group["tmean"].transform("mean")
meta.columns = ["zcta", "nsrdb_id", "lat", "lon", "zcta_centroid",
                "utc_zone", "tmax", "tmin", "tmean", "tmin_mode", "tmax_mode",
                "tmax_mean", "tmin_mean", "tmin_min", "tmax_max",
                "total_tmean"]
meta.to_csv("/scratch/twillia2/fleet/final_long.csv") # Saves everything

# Now just the summary stats
meta["nsrdb_ids"] = meta.groupby("zcta")["nsrdb_id"].transform(lambda x: str(list(x)))
meta2 = meta[['zcta', 'zcta_centroid', 'utc_zone', 'tmin_mode', 'tmax_mode',
              'tmax_mean','tmin_mean', 'tmax_max', 'tmin_min', 'total_tmean',
              'nsrdb_ids']].drop_duplicates()
meta2["lon"] = meta2["zcta_centroid"].apply(lambda x: x.x)
meta2["lat"] = meta2["zcta_centroid"].apply(lambda x: x.y)
meta2 = meta2.drop("zcta_centroid", axis=1)
meta2 = meta2[['zcta', 'lon', 'lat', 'nsrdb_ids', 'utc_zone', 'tmin_mode',
               'tmax_mode', 'tmin_mean', 'tmax_mean', 'tmin_min', 'tmax_max',
               'total_tmean']]
meta2 = meta2.sort_values("zcta")
meta2 = meta2.reset_index(drop=True)
meta2.to_csv("/scratch/twillia2/fleet/final_short.csv")


meta2['geometry'] = meta2.apply(lambda x: Point(x['lon'], x['lat']), axis=1)
meta2 = gpd.GeoDataFrame(meta2, geometry="geometry")
meta2.to_file("/scratch/twillia2/fleet/final.shp")
# For if we ever manage to get the full timezone-filtered value and time arrays
# zips = meta["zip"].values
# zips = np.array([np.string_(x) for x in zips])
# meta["zip"] = zips
# col_dtypes = {"index": "<i8", "zip": "S8", "resource_index": "<i8",
#               "lat": "<f4", "lon": "<f4", "zone": "<f8"}
# df = meta.to_records(column_dtypes=col_dtypes)

# # Save time and temperature
# with hp.File("/scratch/twillia2/fleet/temperature_daytime.h5", "w") as ds:
#     ds.create_dataset(name="meta", data=df)
#     ds.create_dataset(name="temperature", data=temp_filtered)
#     ds.create_dataset(name="local_date", data=time_filtered)
#     ds.attrs["description"] = ("This contains air temperature from the NSRDB" +
#                                "within US zip codes between 8:00 AM and " +
#                                "5:00 PM local time.")
