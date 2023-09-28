#main code
import ee
from pprint import pprint
from react.ReactTool import React

ee.Initialize()

# Example of Shire River, Malawi

## set region by coordinates
# xx, yy, XX, YY = 35.216675,-17.122417,35.311432,-16.945156
# xx, yy, XX, YY = 35.216675,-17.122417,35.255127,-17.063349
# xx, yy, XX, YY = 35.226288,-14.887723,35.345078,-14.754299
xx, yy, XX, YY = 35.267487,-14.888387,35.316582,-14.817039
region    = ee.Geometry.Rectangle([xx, yy, XX, YY])

# set region by GEE asset
# asset = ee.FeatureCollection('users/deltareseflows/tenryu-IAHR')
# region = asset.geometry()\               

start     = 2021   
end       = 2022
band      ='ndwi'
outputdir = 'Z:\\Projecten\\REACT\\output'
id        = 'ShireRiver_testttt'

re = React()
re.floodFrequency(region,start,end,outputdir,band,id)
# re.morphology(region,start,end,band,outputdir,id)
# re.floodExtent(region,start,end,band,outputdir,id)