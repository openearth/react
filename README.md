# REACT documentation

The REACT tool aims to rely on open-source, publicly available datasets to conduct rapid environmental flows assessment to enable timely stakeholder involvement. This is a repository containing scripts or tools used for REACT, the Deltares tool for environmental flows assessment at multiple spatial scales.

## Getting started
First, install the environment.
```
conda env create -f react_environment.yml
```

Next, ensure you have a google earth engine account. If not, sign-up [here](https://code.earthengine.google.com/register). You will need it to authenticate/initialize earth engine packages within the tool. For more information on how to Authenticate please visit [here](https://developers.google.com/earth-engine/apidocs/ee-authenticate). 

Finally, use ```example.py``` to run the script with your own region examples. A section of the Pineios River in Greece is used as a default example, but feel free to change it with your own defined geometries from e.g. a shapefile or GeoJSON file.

## User guide
This document describes the functions implemented in REACT using Google Earth Engine resources. REACT tools produce Earth Observations information on 1) flood frequency, 2) flood extent, and 3) river morphology. Below you can find a general description of each tool and details of the inputs and outputs of the tool.
## Description of tools 
### Flood frequency
The flood frequency tool provides insights into the observed floods in a region. This tool produces a grided map of a selected region where each grid or pixel represents a number between 0 and 1 where 0 indicates any flood and 1 a region that was flooded during the total number of available observations during a selected year. Values between 0 and 1 represent regions where the floods were identified in some images (or events during the year). The number of available observations is determined by the number of images of the Landsat 4, Landsat 5, Landsat 8, Landsat 9, and Sentinel 2 collections without cloud coverage in the period of interest.  
### Flood extent
The flood extent tool provides insights into the spatial extension of the flooded areas in a region per year. This tool produces a grided map of a selected region where each grid or pixel represents whether 0 or 1, where 0 indicates any flood and 1 is a region that was flooded during the year. The tool derives the flooded areas over an annual mean composite image. The annual mean composite image portrays the mean performance of reflectance in each pixel, so the annual flood extent maps portray the mean flood extent. The annual mean composite image is computed with the images of the Landsat 4, Landsat 5, Landsat 8, Landsat 9, and Sentinel 2 collections without cloud coverage in the period of interest.  The flood extent tool produces additionally a plot and a table that indicate the number of pixels that are identified as flooded areas per year in the region and period of interest. 
### River morphology
The river morphology tool provides insides of the erosion and accretion of the riverbanks. This tool produces a grided map of a selected region where each grid or pixel represents whether -1, 0 or 1, where -1 indicates accretion, 0 any change, and 1 erosion. The maps are produced by subtracting the flood extent map of a year of interest from the flood extent map of the year before. If a pixel was flooded one year of interest and was not the year before, a value of 1 will be shown in the river morphology map representing erosion of the river. If a pixel was not flooded one year of interest and was one year before, a value of -1 will be shown in the river morphology map representing the accretion of the river. If no change happened between the year of interest and the year before, then a value of 0 will be shown in the river morphology map representing no change in the river. The maps are computed every year and between the start and end year of the analysis. 
## Description of inputs
The three tools described above require the same inputs. They can be used and run independently of each other. Find below the description of each input.

* Region [ee.Geometry].- The tool requires a geometry that can be read by Google Earth Engine. The example.py code provides a way to create a geometry based on a rectangle. To run the tool with this method the user must provide the minimum and maximum corners of the rectangle as a list of two points each in the format of GeoJSON 'Point' coordinates or a list of four numbers in the order xMin, yMin, xMax, yMax. The coordinates of the corners must be in the projection EPSG:4326. Other ways to provide create geometries with other attributes can be found at [here] https://developers.google.com/earth-engine/guides/geometries.

* Start [integer].- Start year of the period of interest. The year must be provided in YYYY format and can start from 1982 to 2022.

* End [integer].- End year of the period of interest. The year must be provided in YYYY format and must be at least one year ahead of the selected start year. 

* Band [string].- Whether “ndwi” or “mndwi”. The selected band is used to identify flooded areas with the Otsu’s method.

* Outputdir [path or string].- Local directory to save the outputs of the tools. The output must be a path object, or a string of the path preceded by “r” to ignore the whitespace characters (see the example.py code).

* Id [string].- Id of the output files. The Id will precede the name of each output file and must be given as a string.

## Description of outputs
### Flood frequency
* Id_available_pixels_over_YYYY.tiff: Geotiff file with the number of available images per grid in the year YYYY. 
* Id_flooded_pixels_over_YYYY.tiff: Geotiff file with the number of flood events per grid in the year YYYY.
* Id_flood_frequency_2023.tiff: Geotiff file resulting from the division between the number of flood events and the number of available images per grid in the year YYYY.
### Flood extent
* Id_flood_extent_per_annual_mean_composite_image_2023.tiff: Geotiff file with 0 and 1 values representing the flooded areas in the YYYY.
* Id_number_of_flooded_pixels_per_annual_mean_composite_image.csv: Tabular file with the number of flooded pixels per year in the period of interest.
* Id_number_of_flooded_pixels_per_annual_mean_composite_image.html: Plot with the number of flooded pixels per year in the period of interest.
### River morphology
* Id_morphology_YYYY.tiff: Geotiff file resulting from the subtraction between the flood extent of a year YYYY and the flood extent of a year before representing -1 for the accretion, 0 for no change, and 1 for erosion in rivers in a year.
* Id_morphology_yyyy_YYYY.tiff: Geotiff file resulting from the subtraction between the flood extent of the end year YYYY and -the flood extent of the start year yyyy of a period of analysis, representing -1 for the accretion, 0 for no change, and 1 for erosion in rivers in a period of analysis.

## Contact
For further enquiries regarding the use of the software, please approach the following developers: Mario Fuentes Monjaraz (Mario.FuentesMonjaraz@deltares.nl) and Robyn Gwee (Robyn.Gwee@deltares.nl). The REACT tool has been developed with support from the Deltares Water Resources Strategic Research Programme.
