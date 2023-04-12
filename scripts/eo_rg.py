import hydrafloods as hf
import ee, geemap
from eepackages import assets
Map = geemap.Map(center=(34.7, 137.8), zoom=11)

ee.Initialize()

start = "1990-01-01"
end = "1995-01-01"
region = ee.Geometry.Rectangle([137.767181,34.641682,137.842026,34.878045])

assetsLoc = 'users/robynsimingwee/ee-react/'

def getNDWI(img):
    ndwi = img.addBands(
        img.normalizedDifference(['green', 'nir']).select('nd').rename('ndwi')
        )
    return ndwi

col = assets.getMostlyCleanImages(
    assets.getImages(
    region,
    {
        'missions': ['L4', 'L5', 'L8', 'L9', 'S2'],
        'cloudMask': True,
        'filter': ee.Filter.date(start, end),
        'filterMaskedFraction': 0.9,
    }
    ),
    region,
    {
    'percentile': 90,
    'qualityBand': 'blue'
    }
).map(lambda f: f.clip(region)).map(getNDWI)


Map.addLayer(col.first().select('ndwi')); Map

def applyOtsu(img):
    return hf.edge_otsu(
        img,
        region=region,
        band='ndwi',
        thresh_no_data=-0.2,
        # edge_buffer=300,
        invert=True
    )

water_viz = {
    "min":0,
    "max":1,
    "palette":"silver,navy",
    "region":region,
    "dimensions":2000
}
print(applyOtsu(col.first()).getThumbURL(water_viz))
# Map.addLayer(applyOtsu(col.first())); Map

