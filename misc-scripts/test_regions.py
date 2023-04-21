# parameterisation
xx, yy = -19.864248, 34.455185
XX, YY = -19.407021, 34.892578

import hydrafloods as hf
import ee, geemap, urllib
from eepackages import assets

ee.Initialize()
import json

with open('D:\\react\\data\\beira\\beira.geojson') as f:
    data = json.load(f)

coords = data['features'][0]['geometry']['coordinates']
region = ee.Geometry.Polygon(coords[0])

#region = ee.Geometry.Rectangle([yy,xx,YY,XX])


def getndwi(img):
    ndwi = img.addBands(
        img.normalizedDifference(['green', 'nir']).select('nd').rename('ndwi')
        )
    return  ndwi

def getmndwi(img):
    mndwi = img.addBands(
        img.normalizedDifference(['green', 'swir']).select('nd').rename('mndwi')
        )
    return  mndwi


def getcollection(ts, te, region):
    
    col = assets.getMostlyCleanImages(
        assets.getImages(
        region,
        {
            'missions': ['L4', 'L5', 'L8', 'L9', 'S2'],
            'cloudMask': True,
            'filter': ee.Filter.date(ts, te),
            'filterMaskedFraction': 0.9,
        }
        ),
        region,
        {
        'percentile': 90,
        'qualityBand': 'blue'
        }
    ).map(lambda f: f.clip(region)).map(getndwi).map(getmndwi)

    return col

def hydroperiod(col, region, band):

    def applyOtsu(img):
        otsu = hf.edge_otsu(
            img,
            region=region,
            band=band,
            thresh_no_data=-0.2,
            # edge_buffer=300,
            invert=True)
        
        return img.addBands(otsu.select('water'))
    
    otsu_col       = col.map(applyOtsu).select('water')
    otsu_col_sum   = otsu_col.sum().rename('water_sum').select('water_sum')
    otsu_col_count = otsu_col.count().rename('water_count').select('water_count')
    overall        = otsu_col.mean().addBands([otsu_col_sum, otsu_col_count])

    bands_for_expressions = {
                            'water_sum'  : overall.select('water_sum'),
                            'water_count': overall.select('water_count'),
                            }
    
    frequency = overall.expression('water_sum/water_count',bands_for_expressions).rename("frequency")
    return otsu_col_sum, otsu_col_count, frequency

def downloadhydroperiodperyear(region,startyear,endyear,outputdir, id, band):  
    for yr in range(startyear,endyear+1):
        ts = ee.Date.fromYMD(yr, 1, 1)
        te = ts.advance(1, 'year')
        col_filtered = getcollection(ts, te, region)
        print(col_filtered.size().getInfo())
        sum, count, frequency = hydroperiod(col_filtered, region, band)

        name_file_frequency = "frequency_maps_"+ str(yr) +'_'+ id
        name_file_sum       = "sum_maps_"+ str(yr) +'_'+ id
        name_file_count     = "count_maps_"+ str(yr) +'_'+ id

        def downloadurl(img, name_file,band):
            file_path_tiff = outputdir + name_file + ".tif"
            urlD = img.getDownloadUrl({
                'bands' : band,
                'region': region,
                'scale' : 10,
                'format': 'GEO_TIFF'})
            
            try:
                urllib.request.urlretrieve(urlD, file_path_tiff)
                print(f"{name_file} has been downloaded successfully.")
            except:
                print(f"Failed to download {name_file}.")
            return  
        
        downloadurl(frequency, name_file_frequency,'frequency')
        downloadurl(sum,name_file_sum,'water_sum')
        downloadurl(count, name_file_count,'water_count')
    return

start     = 2020
end       = 2021
band      ='mndwi'

outputdir = 'D:\\react\\data\\beira'

Map = geemap.Map(center=((xx+XX)/2,(yy+YY)/2), zoom=11)
Map.addLayer(region); Map

downloadhydroperiodperyear(
    region, 
    start,
    end,
    outputdir,
    'v1', 
    band)

