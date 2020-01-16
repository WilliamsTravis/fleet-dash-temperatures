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
import h5py as hp
import json
import pandas as pd

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
meta = append_zip_tz(crddf)

# Now, subset the temperature data by the new resource index
resource_index = meta["resource_index"].values
resource = temp[:, resource_index]

# Append summary statistics for values between 8 AM and 5 PM, local time
meta = day_stats(meta=meta, resource=resource, time_index=time_index,
                 hour_range=(7, 17))
ds.close()

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
meta.columns = ["zcta", "nsrdb_index", "lat", "lon", "zcta_centroid",
                "utc_zone", "tmax", "tmin", "tmean", "tmin_mode", "tmax_mode",
                "tmax_mean", "tmin_mean", "tmin_min", "tmax_max", "total_tmean"]
meta.to_csv("/scratch/twillia2/fleet/final_long.csv") # Saves everything

# Now just the summary stats
meta["nsrdb_indices"] = meta.groupby("zcta")["nsrdb_index"].transform(lambda x: str(list(x)))
meta2 = meta[['zcta', 'zcta_centroid', 'utc_zone', 'tmin_mode', 'tmax_mode',
              'tmax_mean','tmin_mean', 'tmax_max', 'tmin_min', 'total_tmean',
              'nsrdb_indices']].drop_duplicates()
meta2["lon"] = meta2["zcta_centroid"].apply(lambda x: x[0])
meta2["lat"] = meta2["zcta_centroid"].apply(lambda x: x[1])
meta2 = meta2.drop("zcta_centroid", axis=1)
meta2 = meta2[['zcta', 'lon', 'lat', 'nsrdb_indices', 'utc_zone', 'tmin_mode',
               'tmax_mode', 'tmin_mean', 'tmax_mean', 'tmin_min', 'tmax_max',
               'total_tmean']]
meta2 = meta2.sort_values("zcta")
meta2 = meta2.reset_index(drop=True)
meta2.to_csv("/scratch/twillia2/fleet/final_short.csv")



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
