import hydrafloods as hf
import ee, os
from eepackages import assets
import urllib.request
import pandas as pd
import altair as alt

class React:
    """_summary_
    """    
    def __init__(self):
        self.__help__ = '''
        Description
        '''
        self.__version__ = "Vesion"

    def getNDWI(self,img):
        ndwi = img.addBands(
            img.normalizedDifference(['green', 'nir']).select('nd').rename('ndwi')
            )
        return  ndwi

    def getMNDWI(self,img):
        mndwi = img.addBands(
            img.normalizedDifference(['green', 'swir']).select('nd').rename('mndwi')
            )
        return  mndwi

    def getCollection(self, ts, te, region):
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
        ).map(lambda f: f.clip(region)).map(self.getNDWI).map(self.getMNDWI)
        return col

    def annualOtsu(self,col_mean, region, band):
        def applyOtsu(img):
            otsu = hf.edge_otsu(
                img,
                region=region,
                band=band,
                thresh_no_data=-0.2,
                # edge_buffer=300,
                invert=True)
            return otsu
        otsu_col        = applyOtsu(col_mean)
        otsu_col_annual = otsu_col.rename('water_annual').select('water_annual')
        otsu_col_annual_nonzero = otsu_col_annual.where(otsu_col_annual.eq(0), 0.0000000001)
        return otsu_col_annual_nonzero

    def downloadUrl(self, img, name_file,band,outputdir,region):
        file_path_tiff = os.path.join(outputdir, f'{name_file}.tiff')
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

    def plotfloodextent(self, df, outputdir,id):
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
        chart.save(outputdir + "/" + id + "_number_of_flooded_pixels_per_annual_mean_composite_image.html")
        return 

    def floodExtent(self,region,startyear,endyear,band,outputdir, id):
        print('-----------Flood extent analysis----------')
        image_year = []
        pixel_image_list = []
        for yr in range(startyear,endyear+1):
            # create a collection within the time start(ts) and time end(te)
            ts = ee.Date.fromYMD(yr, 1, 1)
            te = ts.advance(1, 'year')
            col_filtered = self.getCollection(ts, te, region).select(band).mean()    #apply mean and then Otsu
            print("Year of analysis: "+ str(yr))
            
            for band in col_filtered.bandNames().getInfo():
                print("Band to use: "+ band)

            # apply otsu threshold over annual collection
            annual_otsu = self.annualOtsu(col_filtered, region, band)
            
            name_file_flood_extent = id + "_flood_extent_per_annual_mean_composite_image_"+ str(yr)
            self.downloadUrl(annual_otsu, name_file_flood_extent,'water_annual', outputdir,region)

            # count number of flooded pixels per year
            reducer     = ee.Reducer.sum()
            pixels_sum  = annual_otsu.select(['water_annual']).reduceRegion(reducer= reducer,geometry=region,scale=30,maxPixels=1e9).getInfo()
            image_year.append(str(yr)+'-01-01')
            pixel_image_list.append(pixels_sum['water_annual'])
            print('\n')

        # create table of number of flooded pixels per year
        table    = {'Year':image_year,'Pixels':pixel_image_list}
        df = pd.DataFrame(table).set_index('Year') 
        df.to_csv(os.path.join(outputdir, f'{id}_floodextent.csv')) 
        self.plotfloodextent(df, outputdir,id)
        return 

    def morphology(self,region,startyear,endyear,band,outputdir, id): 
        print('-----------Morphology analysis----------')
        print('-Flood extent estimation:')
        print('\n')
        images_annual_otsu = []
        for yr in range(startyear,endyear+1):
            # create a collection within the time start(ts) and time end(te)
            ts = ee.Date.fromYMD(yr, 1, 1)
            te = ts.advance(1, 'year')
            col_filtered = self.getCollection(ts, te, region).select(band).mean()

            print("Year of analysis: "+ str(yr))
            for band in col_filtered.bandNames().getInfo():
                print("Band to use: "+ band)

            globals()['annualotsu_%s' %str(yr)] = self.annualOtsu(col_filtered, region, band)

            # name_file_erotion = id + "_flood_extent_annual_composite_image_"+ str(yr) 
            # self.downloadUrl(globals()['annualotsu_%s' %str(yr)], name_file_erotion,'water_annual', outputdir,region)

            images_annual_otsu.append(globals()['annualotsu_%s' %str(yr)])
            print('\n')

        print('-Erosion acretion estimation:')
        print('\n')
        for yr, ind in zip(range(startyear,endyear),range(len(images_annual_otsu)-1)):
            diff_image = images_annual_otsu[ind+1].subtract(images_annual_otsu[ind])
            print("Year of analysis: "+ str(yr+1))
            for band in diff_image.bandNames().getInfo():
                print("Band to use: "+ band)
            diff_image = diff_image.rename('water_difference').select('water_difference')
            name_file_erotion = id +"_morphology_"+ str(yr+1) 
            self.downloadUrl(diff_image, name_file_erotion,'water_difference', outputdir,region)
            print('\n')
        return 

    def annualFloodFrequency(self,col, region, band):
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

    def floodFrequency(self,region,startyear,endyear,band,outputdir,id):  
        print('-----------Flood Frequency----------')
        for yr in range(startyear,endyear+1):
            # create a collection within the time start(ts) and time end(te)
            ts = ee.Date.fromYMD(yr, 1, 1)
            te = ts.advance(1, 'year')
            col_filtered = self.getCollection(ts, te, region)
            
            print("Year of analysis: "+ str(yr))
            print("Number of images: "+ str(col_filtered.size().getInfo()))
            
            # get the annual flood frequency
            sum, count, frequency = self.annualFloodFrequency(col_filtered, region, band)

            name_file_frequency = id + "_flood_frequency_"+ str(yr)
            name_file_sum       = id + "_flooded_pixels_over_"+ str(yr)
            name_file_count     = id + "_available_pixels_over_"+ str(yr)

            self.downloadUrl(frequency, name_file_frequency,'frequency',outputdir,region)
            self.downloadUrl(sum,name_file_sum,'water_sum',outputdir,region)
            self.downloadUrl(count, name_file_count,'water_count',outputdir,region)
            print('\n')
        return
