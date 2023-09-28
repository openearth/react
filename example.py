#main code
import ee
from react.ReactTool import React

ee.Initialize()

# Example of Mahanadi River, India
xx, yy, XX, YY = 85.125504,20.321448,85.469513,20.419935

region    = ee.Geometry.Rectangle([xx, yy, XX, YY])
start     = 2020
end       = 2021
band      ='ndwi'
outputdir = r'..//data//'
id        = 'test-mahanadi-ndwi'

re = React()
re.floodFrequency(region,start,end,outputdir,band,id)
re.morphology(region,start,end,band,outputdir,id)
re.floodExtent(region,start,end,band,outputdir,id)
