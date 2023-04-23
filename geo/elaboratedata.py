import xarray as xr
import matplotlib.pyplot as plt
datafile = "outputtest.grib"
ds = xr.open_dataset("outputtest.grib")

df = ds.to_dataframe()
df.head()
df.describe()
