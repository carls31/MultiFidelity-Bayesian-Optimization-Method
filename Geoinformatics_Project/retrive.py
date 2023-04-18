import cdsapi, json 

with open("request.json") as req:
    request = json.load(req)

cds = cdsapi.Client(request.get("url"), request.get("uuid") + ":" + request.get("key")) 

#year = ["20{:02d}".format(i) for i in range(0,10)]
year = ["2004"]

days = ["{:02d}".format(i) for i in range(0,32)]
times = ["{:02d}:00".format(i) for i in range(0,24)]
month = ["{:02d}".format(i) for i in range(1,24)]

print(year)

area= "6.002/33.501/-5.202/42.283"
#'variable': '2m_temperature',
  
r = cds.retrieve("reanalysis-era5-pressure-levels",
{
"variable": "temperature",
"pressure_level": "1000",
"product_type": "reanalysis",
"year": year,
"month": month,
"day": days,
"time": times,
"area":['75', '-15', '30', '42.5'], #europe https://cds.climate.copernicus.eu/toolbox/doc/how-to/1_how_to_retrieve_data/1_how_to_retrieve_data.html
"format": "netcdf"
}, )
r.download("downloadERA-1yearEurope.nc")

'''
import xarray as xr 
ds = xr.open_dataset("downloadERA-1yearEurope.nc")
ds.variables

ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby('longitude')
'''