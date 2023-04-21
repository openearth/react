#main code
import json
import ee
from ReactTool import React

ee.Initialize()

with open('C:/Users/fuentesm/CISNE/REACT/Maps/River_mouth.geojson') as f:
    data = json.load(f)

coords    = data['features'][0]['geometry']['coordinates']
region    = ee.Geometry.Polygon(coords[0])
start     = 2000
end       = 2002
band      ='mndwi'
outputdir = 'C:/Users/fuentesm/CISNE/REACT/Maps/'
id        = 'A'
re = React()
re.floodFrequency(region,start,end,outputdir,band,id)
re.morphology(region,start,end,band,outputdir,id)
re.floodExtention(region,start,end,band,outputdir,id)