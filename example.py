#main code
import ee
from react.ReactTool import React

ee.Initialize()

# Example of Lower Tenryu, Japan
xx, yy, XX, YY = 137.774734,34.647896,137.838593,34.871848


region    = ee.Geometry.Rectangle([xx, yy, XX, YY])
start     = 2015
end       = 2022
band      ='ndwi'
outputdir = r'D:\react\data\lower-tenryu'
id        = 'lower-tenryu-ndwi'

re = React()
re.floodFrequency(region,start,end,band,outputdir,id)
re.morphology(region,start,end,band,outputdir,id)
re.floodExtent(region,start,end,band,outputdir,id)
