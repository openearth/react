import hydrafloods as hf
import ee, geemap
import folium
from folium import plugins
from eepackages import assets
from eepackages import utils
import urllib.request
import pandas as pd
import altair as alt
import openpyxl

ee.Initialize()

class React:
    """_summary_
    """    
    def __init__(self):
        self.__help__ = '''
        Description
        '''
        self.__version__ = "Vesion"
    
    def getndwi(self,img):
        ndwi = img.addBands(
            img.normalizedDifference(['green', 'nir']).select('nd').rename('ndwi')
            )
        return  ndwi

    def getmndwi(self,img):
        mndwi = img.addBands(
            img.normalizedDifference(['green', 'swir']).select('nd').rename('mndwi')
            )
        return  mndwi
    
    def getcollection(self,ts, te, region):
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
        ).map(lambda f: f.clip(region)).map(self.getndwi).map(self.getmndwi)
        return col

    def downloadurl(self,img, name_file,band, outputdir, region):
        file_path_tiff = outputdir + name_file + ".tiff"
        urlD = img.getDownloadUrl({'bands' : band,
                                        'region': region,
                                        'scale' : 30,
                                        'format': 'GEO_TIFF'})
        try:
            urllib.request.urlretrieve(urlD, file_path_tiff)
            print(f"{name_file} has been downloaded successfully.")
        except:
            print(f"Failed to download {name_file}.")
        return  
    
    def plotfloodextent(self,df, outputdir, id):
        dff = df.reset_index()
        chart = alt.Chart(dff).mark_bar(size=1).encode(
            x='Year:T',
            y='Pixels:Q',
            color=alt.Color(
                'Pixels:Q', scale=alt.Scale(scheme='redyellowgreen', domain=(300000, 350000))),
            tooltip=[
                alt.Tooltip('Year:T', title='Date'),
                alt.Tooltip('Pixels:Q', title='Pixels')
            ]).properties(width=600, height=300)
        chart.save(outputdir + "/" + id + "_GPP_daily.html")
        return
    
    def hydroperiod(self, col, region, band):
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
    
    def downloadhydroperiodperyear(self, region,startyear,endyear,outputdir, band, id):  
        for yr in range(startyear,endyear+1):
            ts = ee.Date.fromYMD(yr, 1, 1)
            te = ts.advance(1, 'year')
            col_filtered = self.getcollection(ts, te, region)
            print(col_filtered.size().getInfo())
            sum, count, frequency = self.hydroperiod(col_filtered, region, band)

            name_file_frequency = "frequency_maps_"+ str(yr) +'_'+ id
            name_file_sum       = "sum_maps_"+ str(yr) +'_'+ id
            name_file_count     = "count_maps_"+ str(yr) +'_'+ id

            self.downloadurl(frequency, name_file_frequency,'frequency',outputdir,region)
            self.downloadurl(sum,name_file_sum,'water_sum',outputdir,region)
            self.downloadurl(count, name_file_count,'water_count',outputdir,region)
        return
    
    def erotionacretion(self,region,startyear,endyear,band,outputdir, id): 
        images_annual_otsu = []
        for yr in range(startyear,endyear+1):
            ts = ee.Date.fromYMD(yr, 1, 1)
            te = ts.advance(1, 'year')
            col_filtered = self.getcollection(ts, te, region).select(band).mean()
            for band in col_filtered.bandNames().getInfo():
                print(band)
            globals()['annualotsu_%s' %str(yr)] = self.annualotsu(col_filtered, region, band)

            name_file_erotion = "annual_otsu_erosion_accretion_"+ str(yr+1) +'_'+ id
            self.downloadurl(globals()['annualotsu_%s' %str(yr)], name_file_erotion,'water_annual', outputdir,region)

            images_annual_otsu.append(globals()['annualotsu_%s' %str(yr)])

        for yr, ind in zip(range(startyear,endyear),range(len(images_annual_otsu)-1)):
            print(yr+1) 
            diff_image = images_annual_otsu[ind+1].subtract(images_annual_otsu[ind])
            for band in diff_image.bandNames().getInfo():
                print(band)
            diff_image = diff_image.rename('water_difference').select('water_difference')
            name_file_erotion = "erosion_accretion_"+ str(yr+1) +'_'+ id
            self.downloadurl(diff_image, name_file_erotion,'water_difference', outputdir,region)
        return 
    
    def floodextension(self,region,startyear,endyear,band,outputdir, id):
        image_year = []
        pixel_image_list = []
        for yr in range(startyear,endyear+1):
            ts = ee.Date.fromYMD(yr, 1, 1)
            te = ts.advance(1, 'year')
            col_filtered = self.getcollection(ts, te, region).select(band).mean()
            for band in col_filtered.bandNames().getInfo():
                print(band)
            print(yr)
            annual_otsu = self.annualotsu(col_filtered, region, band)
            name_file_flood_extent = "annual_otsu_flood_extent_"+ str(yr+1) +'_'+ id
            self.downloadurl( annual_otsu, name_file_flood_extent,'water_annual', outputdir,region)

            reducer     = ee.Reducer.sum()
            pixels_sum  = annual_otsu.select(['water_annual']).reduceRegion(reducer= reducer,geometry=region,scale=10,maxPixels=1e9).getInfo()
            image_year.append(str(yr)+'-01-01')
            pixel_image_list.append(pixels_sum['water_annual'])
        table    = {'Year':image_year,'Pixels':pixel_image_list}
        df = pd.DataFrame(table).set_index('Year')
        df.to_csv(outputdir + "/" + id + "_flood_extend.csv") 
        self.plotfloodextent(df, outputdir)
        return 
    

#main code
import json

with open('C:/Users/fuentesm/CISNE/REACT/Maps/River_mouth.geojson') as f:
    data = json.load(f)

coords    = data['features'][0]['geometry']['coordinates']
region    = ee.Geometry.Polygon(coords[0])
start     = 2020
end       = 2021
band      ='mndwi'
outputdir = 'C:/Users/fuentesm/CISNE/REACT/Maps/'
re = React()
# re.downloadhydroperiodperyear(region,start,end,outputdir, band, 'v7')
# re.erotionacretion(region,start,end,outputdir, band, 'v7')
re.floodextension(region,start,end,outputdir, band, 'v7')