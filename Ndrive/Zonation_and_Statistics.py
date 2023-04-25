# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 14:56:41 2022

Test zonation based on wflow results

@author: oorschot
"""
import os
import netCDF4 as nc
from netCDF4 import Dataset
import pandas as pd
import xarray as xr
import numpy as  np
from os.path import join, dirname
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm, colors
import hydromt
# plot maps dependencies
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import descartes # required to plot polygons
import cartopy.io.img_tiles as cimgt
import datetime as datetime

#%% set paths
path = r"n:\Projects\11208000\11208005\F. Other information\025- EFlows\Manual\Tutorial files\wflow_model_updated"
root = r"n:\Projects\11208000\11208005\F. Other information\025- EFlows\Manual\Tutorial files\wflow_model_updated"
nc_name = 'output.nc'
nc_file  = os.path.join(path, 'run_default', nc_name) # netcdf file met wflow output
mod = hydromt.WflowModel(root, mode='r')
output =r"n:\Projects\11208000\11208005\F. Other information\025- EFlows\Manual\Tutorial files\wflow_model_updated\output"
AllDataPath = os.path.join(output, 'Output.csv')

#%% user defined parameters
timestepsecs = '86400'  # timestep in seconds. see *toml file in wflow model folder
refD = '19000101' # reference date. see *toml file in wflow model folder
start_year = 2009 # manually enter start year of statistics calculation, this should be a year for which discharges are available for the whole year
num_years = 1 # manually enter the amount of years which are completely available
write_output = True # schrijf output van flow indicators weg per jaar


#%% inspect data and variables
ds = nc.Dataset(nc_file)
print(ds)
print(ds.__dict__)
for dim in ds.dimensions.values():
    print(dim)
for var in ds.variables.values():
    print(var)   
 
    
reference_date = datetime.datetime(int((refD[0:4])), int((refD[4:6])), int((refD[6:8])))        
# determine start end end dates
time = ds['time'][:]  
startdate = reference_date +datetime.timedelta(days=time[0])
enddate = reference_date +datetime.timedelta(days=time[len(time)-1]) 
timestep = timestepsecs + 'S' 
date_range = pd.date_range(start = startdate, end = enddate, freq = timestep)   

startDmdu = time[0]  # start date of simulation with reference to refD
endDmdu = time[len(time)-1]  # end date of simulation with reference to refD
TotalSimTime =   endDmdu - startDmdu 

Dates_df = pd.DataFrame(data=date_range)
Dates_df = Dates_df.rename(columns={0: 'date'}) 
Dates_df['year'], Dates_df['month'], Dates_df['day'] = Dates_df['date'].dt.year, Dates_df['date'].dt.month, Dates_df['date'].dt.day
    
 
#%% extract required data and put in table next to lat long coordinates

# extract coordinates
lat = ds['lat'][:] 
long = ds['lon'][:] 
lat = pd.DataFrame(data=lat).iloc[::-1]    


cors = []
# create list of lat long coordinates in vector format
for i in range(len(long)):
    df1 = pd.DataFrame(lat)
    df2 = long[i]
    df2 = pd.DataFrame(np.repeat(df2, len(lat), axis=0))
    frames = [df1, df2]
    data = pd.concat(frames, axis = 1, ignore_index = True)    
    data.reset_index(drop=True, inplace=True)
    cors.append(data)
final_cors = pd.concat(cors, axis = 0, ignore_index = True)  

# add digital elevation, river slope, river width, river height and stream order to coordinates

# extract DEM data for all coordinates
da =mod.staticmaps['wflow_dem'].values
data_step = pd.DataFrame(data=da)
dem_data = []
for i in range(len(long)):
    df3 = pd.DataFrame(data_step[i])
    frames = [df3]
    data = pd.concat(frames, axis = 1, ignore_index = True)    
    data.reset_index(drop=True, inplace=True)
    dem_data.append(data)
dem_complete = pd.concat(dem_data, axis = 0, ignore_index = True)      
    
# extract river slope data for all coordinates   
da =mod.staticmaps['RiverSlope'].values
data_step = pd.DataFrame(data=da)
slope_data = []
for i in range(len(long)):
    df3 = pd.DataFrame(data_step[i])
    frames = [df3]
    data = pd.concat(frames, axis = 1, ignore_index = True)    
    data.reset_index(drop=True, inplace=True)
    slope_data.append(data)
slope_complete = pd.concat(slope_data, axis = 0, ignore_index = True)     
    

# extract river width data for all coordinates
da =mod.staticmaps['wflow_riverwidth'].values
data_step = pd.DataFrame(data=da)
river_width_data = []
for i in range(len(long)):
    df3 = pd.DataFrame(data_step[i])
    frames = [df3]
    data = pd.concat(frames, axis = 1, ignore_index = True)    
    data.reset_index(drop=True, inplace=True)
    river_width_data.append(data)
river_width_complete = pd.concat(river_width_data, axis = 0, ignore_index = True)   

# extract river width data for all coordinates
da =mod.staticmaps['wflow_streamorder'].values
data_step = pd.DataFrame(data=da)
stream_order_data = []
for i in range(len(long)):
    df3 = pd.DataFrame(data_step[i])
    frames = [df3]
    data = pd.concat(frames, axis = 1, ignore_index = True)    
    data.reset_index(drop=True, inplace=True)
    stream_order_data.append(data)
stream_order_complete = pd.concat(stream_order_data, axis = 0, ignore_index = True)   

#%% calculate flow regime indicators
# calculate hydrodynamic statistics according to gurnell et al(2004) REFORM Framework 2.1

# create lookup table for flood timing
months = pd.DataFrame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12])
startday = pd.DataFrame([1, 32, 61, 92, 122, 153, 183, 214, 245, 275, 306, 336])
frames = [months, startday]
time_table = pd.concat(frames, axis = 1, ignore_index = True)
time_table.rename(columns={0: 'fldtime', 1: 'startday'}, inplace=True)   


qr = ds['q_river'][:,:,:]   # extract river discharge

Q_indicators = []
for y in range(num_years): # loop over years
    cur_year = start_year+y
    temp_year_store_BFI = [] # temporal dataframe to store annual data
    temp_month_store = [] # store montly ratios in lat long format
    
    # extract discharge per year    
    year_index = Dates_df.index[(Dates_df['year'] == cur_year)]
    dates = Dates_df[(Dates_df['year'] == cur_year)]# extract dates per year    
    dates.reset_index(inplace = True, drop = True)# reset index for later concatenation 
    Q_year = qr[year_index] # extract discharges of this year
    
    n_zeros = np.count_nonzero(Q_year==0, axis=0) # count number of zero flow days for each grid cell over the year
    # turn dataframe upside down to match the static data..?
    n_zeros = pd.DataFrame(data=n_zeros).iloc[::-1]    
    
    size = np.shape(Q_year)
    Q_daily_mean = np.mean(Q_year, axis = 0) # annual mean daily discharge per cell
    # turn dataframe upside down to match the static data..?
    Q_daily_mean = pd.DataFrame(data=Q_daily_mean).iloc[::-1]    
    
    Q_daily_stdev = np.std(Q_year, axis = 0)# standard deviation of mean annual daily discharges
    # turn dataframe upside down to match the static data..?
    Q_daily_stdev = pd.DataFrame(data=Q_daily_stdev).iloc[::-1]      
    
    DAYCV = Q_daily_stdev/Q_daily_mean# coefficient of variation calculated as ratio between stdev and mean 
    
    
    Q_threshold = [] # threshold value for flood comparison
    number_no_flow_days = []
    coef_var = []
    # put flood threshold in right lat-long format
    for i in range(len(long)):
        df3 = pd.DataFrame(Q_daily_mean[i])
        df2 = pd.DataFrame(n_zeros[i])
        df4 = pd.DataFrame(DAYCV[i])
        frames = [df3]
        frames2 = [df2]
        frames3 = [df4]
        data = pd.concat(frames, axis = 1, ignore_index = True)   
        data2 = pd.concat(frames2, axis = 1, ignore_index = True)  
        data3 = pd.concat(frames3, axis = 1, ignore_index = True)  
        data.reset_index(drop=True, inplace=True)
        data2.reset_index(drop=True, inplace=True)
        data3.reset_index(drop=True, inplace=True)
        Q_threshold.append(data)
        number_no_flow_days.append(data2)
        coef_var.append(data3)
    Q_threshold_data = pd.concat(Q_threshold, axis = 0, ignore_index = True)     
    no_flow_days = pd.concat(number_no_flow_days, axis = 0, ignore_index = True)  
    coefficient_variation = pd.concat(coef_var, axis = 0, ignore_index = True)  
    

    data_compare_threshold = []
    daily_discharge_data = []
    for f in range(size[0]): # loop over daily discharges to check for flooding frequency
        Current_day_data = Q_year[f,:,:] # extract daily data
        # turn dataframe upside down to match the static data..?
        Current_day_data = pd.DataFrame(data=Current_day_data).iloc[::-1]
        # loop over lat_long to create right data format
        
        daily_data = [] # store daily data
        for i in range(len(long)):
            df3 = pd.DataFrame(Current_day_data[i])
            frames = [df3]
            data = pd.concat(frames, axis = 1, ignore_index = True)    
            data.reset_index(drop=True, inplace=True)
            daily_data.append(data)
        daily_data_complete = pd.concat(daily_data, axis = 0, ignore_index = True)   
        
        Flood_pres = daily_data_complete>Q_threshold_data # compare daily data with annual mean daily discharge to determine flood presence
        Flood_val = Flood_pres*1 # covert bools into numbers
        data_compare_threshold.append(Flood_val)
        daily_discharge_data.append(daily_data_complete)
        
    Threshold_data = pd.concat(data_compare_threshold, axis = 1, ignore_index = True)    # convert to dataframe
    Daily_discharge_data = pd.concat(daily_discharge_data, axis = 1, ignore_index = True)    # convert to dataframe
    dims = np.shape(Threshold_data)

        
    flood_count = []    
    fld_pred_ind = []
    fld_time_ind = []
    for th in range(dims[0]): # loop over cells to calculate number of floods
        Current_data_cell = Threshold_data.iloc[th,:] # extract threshold data for cell
        Current_data_cell = pd.DataFrame(Current_data_cell)
        #check_zero = int(np.sum(Current_data_cell))
        Current_discharge_cell = Daily_discharge_data.iloc[th,:] # extract discharge data for cell
        Current_discharge_cell = pd.DataFrame(Current_discharge_cell)
        if int(np.sum(Current_data_cell)) == 0: # no floods, put zero in dataframe
             flood_count.append(0)
             fld_pred_ind.append(0)
             fld_time_ind.append(0)
             
        else:                
            # create dataset with floods and corresponding dates
            Data_c = pd.concat([dates, Current_data_cell,Current_discharge_cell],axis = 1, ignore_index=True)                
            Data_c.rename(columns={0: 'date', 1: 'year', 2:'month', 3:'day', 4: 'flood', 5:'Q'}, inplace=True)                
            Data_c['value_grp'] = (Data_c.flood.diff(1) != 0).astype('int').cumsum() # analyse data per gridcell to extract floods and calculate cumsum
    
            cumsum_floods = pd.DataFrame({'BeginDate' : Data_c.groupby('value_grp').date.first(), 'FloodVal' : Data_c.groupby('value_grp').flood.max(), 
            'EndDate' : Data_c.groupby('value_grp').date.last(),'BeginMonth' : Data_c.groupby('value_grp').month.first(),
            'EndMonth' : Data_c.groupby('value_grp').month.last(),'Q_max' : Data_c.groupby('value_grp').Q.max(),
            'Consecutive' : Data_c.groupby('value_grp').size()}).reset_index(drop=True) 
            
            # get the location of the maximum value to extract the day at which it occurs
            idx = Data_c.groupby(['value_grp'])['Q'].transform(max) == Data_c['Q']
            A_test=Data_c[idx]
            cumsum_floods['day_Q_max'] = A_test['day'].reset_index(drop=True)  # add days of max flood
            cumsum_floods['month'] = A_test['month'].reset_index(drop=True)# add date of max flood
            cumsum_floods['date_Q_max'] = A_test['date'].reset_index(drop=True)# add date of max flood
            
            # select only floods = Floodval = 1
            Only_flood_data = cumsum_floods[(cumsum_floods['FloodVal'] == 1)]
            
            # calculate the total number of floods exceeding the threshold of this year for this grid cell (Indicator = FLDFREQ)
            dim = np.shape(Only_flood_data)
            flood_count.append(int(dim[0]/2))
    
            # calculate proportion of floods within each two month window and calculate the max over all windows
            floods_in_months = pd.DataFrame({'Count_floods' : Only_flood_data.groupby('month').Consecutive.count()}) 
            total_floods = np.sum(floods_in_months)
            fldpred_month = []
            for f in range(1,11): # loop over months to calculate FLDPRED
                fl = floods_in_months.loc[f:f+1]
                fldpred = np.sum(fl)/total_floods # calculate flood prediction for two months time window
                fldpred_month.append(fldpred)    
            fldpred = np.max(fldpred_month) #FLDPRED value
            fldpred_dataframe = pd.concat(fldpred_month, axis = 0, ignore_index = True)    # convert to dataframe            
            max_fld_predwindow = fldpred_dataframe.idxmax()+1 # FLDTIME value: this is the number corresponding to the interval of months, e.g. 2 =  feb-mar, 5 = may-june etc.
            
            # convert flood time to actual start date of flood
            stdate = time_table[(time_table['fldtime'] == max_fld_predwindow)]
            
            fld_pred_ind.append(fldpred)
            fld_time_ind.append(int(stdate.startday))           

    for j in (range(12)): # loop over months      
        month_index = Dates_df.index[(Dates_df['year'] == cur_year) & (Dates_df['month'] == j+1)]
        Q_month = qr[month_index] # extract discharges of this month
        min_month = np.min(Q_month,axis =0)
        mean_month = np.mean(Q_month, axis =0)
        ratio_month = min_month/mean_month
        # turn dataframe upside down to match the static data..?
        ratio_month = pd.DataFrame(data=ratio_month).iloc[::-1]
        
        # rearrange data into lat long format
        data_latlong = []
        for i in range(len(long)):
            df_cur_rat = pd.DataFrame(ratio_month[i])
            frames = [df_cur_rat]
            data = pd.concat(frames, axis = 1, ignore_index = True)    
            data.reset_index(drop=True, inplace=True)              
            data_latlong.append(data)        
        data_month = pd.concat(data_latlong, axis = 0, ignore_index = True)           
        data_month.reset_index(drop=True, inplace=True)                       
        temp_month_store.append(data_month)
    temp_year_store_BFI = pd.concat(temp_month_store, axis = 1, ignore_index = True)  
    BFI_year = np.mean(temp_year_store_BFI, axis = 1) # base flow index per grid cell of current year
    
    # todo: save data per year
    frames = [pd.DataFrame(BFI_year)*100,pd.DataFrame(flood_count), no_flow_days, pd.DataFrame(fld_pred_ind), pd.DataFrame(fld_time_ind), pd.DataFrame(coefficient_variation)*100]
    Q_indicators_year = pd.concat(frames, axis = 1, ignore_index = True)  
    Q_indicators_year.rename(columns={0: 'BFI', 1: 'FLDFREQ', 2:'ZERODAY', 3:'FLDPRED', 4: 'FLDTIME', 5: 'DAYCV'}, inplace=True)     
    
    if write_output == True: # if output is true, then write annual data for flow indicators in file
        nameCSV = 'Flow_indicators' + '_year_' + str(y) + '.csv'
        AllDataPath = os.path.join(output, nameCSV)  
        Q_indicators_year.to_csv(AllDataPath, sep=';', mode="w", header=True, index=False)   
        
    Q_indicators.append(Q_indicators_year)
    

# combine all years in a 3d dataframe
Total_Q_indicators = np.dstack([Q_indicators])

#%% # zone selection and calculation of statistics for each zone 
fr = [final_cors, dem_complete, slope_complete, river_width_complete, stream_order_complete]   
All_static_data = pd.concat(fr, axis = 1, ignore_index = True)   
All_static_data.rename(columns={0: 'Y', 1: 'X', 2:'elevation', 3:'slope', 4:'river_width', 5:'stream_order'}, inplace=True)
All_static_data = All_static_data.replace(-9999.0, np.NaN) # replance missing values by NaNs
All_static_data = All_static_data.dropna()# remove rows with Nans
MaximumStreamOrder = np.max(All_static_data.stream_order) # find maximum stream order

# round coordinates to match right location
All_static_data['Y'] = round(All_static_data['Y'],4)
All_static_data['X'] = round(All_static_data['X'],4)

Zones = []

# selection on altitude
# ------------- LOW ALTITUDE -------------------
LowAltitude = All_static_data.loc[(All_static_data['elevation'] <= 200)]

# ----------- Gentle gradient -------------------
LA_GentleGradient = LowAltitude.loc[(LowAltitude['slope'] <= 0.02)]

LA_GG_Max = LA_GentleGradient.loc[(LA_GentleGradient['stream_order'] == MaximumStreamOrder)]
LA_GG_Max['ZoneId'] = 'LA_GG_Max'
Zones.append(LA_GG_Max) # add index to list

LA_GG_Max1 = LA_GentleGradient.loc[(LA_GentleGradient['stream_order'] == MaximumStreamOrder-1)]
LA_GG_Max1['ZoneId'] = 'LA_GG_Max1'
Zones.append(LA_GG_Max1) # add index to list

LA_GG_Max2 = LA_GentleGradient.loc[(LA_GentleGradient['stream_order'] == MaximumStreamOrder-2)]
LA_GG_Max2['ZoneId'] = 'LA_GG_Max2'
Zones.append(LA_GG_Max2) # add index to list

#------------- riffles -----------------------
LA_riffle = LowAltitude.loc[(LowAltitude['slope'] > 0.02) & (LowAltitude['slope'] <= 0.04)]

LA_RI_Max = LA_riffle.loc[(LA_riffle['stream_order'] == MaximumStreamOrder)]
LA_RI_Max['ZoneId'] = 'LA_RI_Max'
Zones.append(LA_RI_Max) # add index to list

LA_RI_Max1 = LA_riffle.loc[(LA_riffle['stream_order'] == MaximumStreamOrder-1)]
LA_RI_Max1['ZoneId'] = 'LA_RI_Max1'
Zones.append(LA_RI_Max1) # add index to list

LA_RI_Max2  = LA_riffle.loc[(LA_riffle['stream_order'] == MaximumStreamOrder-2)]
LA_RI_Max2['ZoneId'] = 'LA_RI_Max2'
Zones.append(LA_RI_Max2) # add index to list

# ----------- steep -----------------------
LA_steep = LowAltitude.loc[(LowAltitude['slope'] > 0.04) & (LowAltitude['slope'] <= 0.1)]

LA_ST_Max = LA_steep.loc[(LA_steep['stream_order'] == MaximumStreamOrder)]
LA_ST_Max['ZoneId'] = 'LA_ST_Max'
Zones.append(LA_ST_Max) # add index to list

LA_ST_Max1 = LA_steep.loc[(LA_steep['stream_order'] == MaximumStreamOrder-1)]
LA_ST_Max1['ZoneId'] = 'LA_ST_Max1'
Zones.append(LA_ST_Max1) # add index to list

LA_ST_Max2 = LA_steep.loc[(LA_steep['stream_order'] == MaximumStreamOrder-2)]    
LA_ST_Max2['ZoneId'] = 'LA_ST_Max2'
Zones.append(LA_ST_Max2) # add index to list


# --------------- very steep ----------------
LA_verysteep = LowAltitude.loc[(LowAltitude['slope'] > 0.1)]

LA_VS_Max = LA_verysteep.loc[(LA_verysteep['stream_order'] == MaximumStreamOrder)]
LA_VS_Max['ZoneId'] = 'LA_VS_Max'
Zones.append(LA_VS_Max) # add index to list

LA_VS_Max1 = LA_verysteep.loc[(LA_verysteep['stream_order'] == MaximumStreamOrder-1)]
LA_VS_Max1['ZoneId'] = 'LA_VS_Max1'
Zones.append(LA_VS_Max1) # add index to list

LA_VS_Max2 = LA_verysteep.loc[(LA_verysteep['stream_order'] == MaximumStreamOrder-2)]
LA_VS_Max2['ZoneId'] = 'LA_VS_Max2'
Zones.append(LA_VS_Max2) # add index to list

# ----------- MID ALTITUDE -------------------
MidAltitude = All_static_data.loc[(All_static_data['elevation'] > 200) & (All_static_data['slope'] <= 800)]

# ----------- gentle gradient ------------------
MA_GentleGradient = MidAltitude.loc[(MidAltitude['slope'] <= 0.02)]
MA_GG_Max = MA_GentleGradient.loc[(MA_GentleGradient['stream_order'] == MaximumStreamOrder)]
MA_GG_Max['ZoneId'] = 'MA_GG_Max'
Zones.append(MA_GG_Max) # add index to list

MA_GG_Max1 = MA_GentleGradient.loc[(MA_GentleGradient['stream_order'] == MaximumStreamOrder-1)]
MA_GG_Max1['ZoneId'] = 'MA_GG_Max1'
Zones.append(MA_GG_Max1) # add index to list

MA_GG_Max2 = MA_GentleGradient.loc[(MA_GentleGradient['stream_order'] == MaximumStreamOrder-2)]
MA_GG_Max2['ZoneId'] = 'MA_GG_Max2'
Zones.append(MA_GG_Max2) # add index to list

#------------- riffles -----------------------
MA_riffle= MidAltitude.loc[(MidAltitude['slope'] > 0.02) & (MidAltitude['slope'] <= 0.04)]

MA_RI_Max = MA_riffle.loc[(MA_riffle['stream_order'] == MaximumStreamOrder)]
MA_RI_Max['ZoneId'] = 'MA_RI_Max'
Zones.append(MA_RI_Max) # add index to list
MA_RI_Max1 = MA_riffle.loc[(MA_riffle['stream_order'] == MaximumStreamOrder-1)]
MA_RI_Max1['ZoneId'] = 'MA_RI_Max1'
Zones.append(MA_RI_Max1) # add index to list
MA_RI_Max2 = MA_riffle.loc[(MA_riffle['stream_order'] == MaximumStreamOrder-2)]
MA_RI_Max2['ZoneId'] = 'MA_RI_Max2'
Zones.append(MA_RI_Max2) # add index to list

# ----------- steep -----------------------
MA_steep = MidAltitude.loc[(MidAltitude['slope'] > 0.04) & (MidAltitude['slope'] <= 0.1)]

MA_ST_Max = MA_steep.loc[(MA_steep['stream_order'] == MaximumStreamOrder)]
MA_ST_Max['ZoneId'] = 'MA_ST_Max'
Zones.append(MA_ST_Max) # add index to list
MA_ST_Max1 = MA_steep.loc[(MA_steep['stream_order'] == MaximumStreamOrder-1)]
MA_ST_Max1['ZoneId'] = 'MA_ST_Max1'
Zones.append(MA_ST_Max1) # add index to list
MA_ST_Max2 = MA_steep.loc[(MA_steep['stream_order'] == MaximumStreamOrder-2)]
MA_ST_Max2['ZoneId'] = 'MA_ST_Max2'
Zones.append(MA_ST_Max2) # add index to list

# --------------- very steep ----------------
MA_verysteep = MidAltitude.loc[(MidAltitude['slope'] > 0.1)]

MA_VS_Max = LA_verysteep.loc[(LA_verysteep['stream_order'] == MaximumStreamOrder)]
MA_VS_Max['ZoneId'] = 'MA_VS_Max'
Zones.append(MA_VS_Max) # add index to list
MA_VS_Max1 = LA_verysteep.loc[(LA_verysteep['stream_order'] == MaximumStreamOrder-1)]
MA_VS_Max1['ZoneId'] = 'MA_VS_Max1'
Zones.append(MA_VS_Max1) # add index to list
MA_VS_Max2 = LA_verysteep.loc[(LA_verysteep['stream_order'] == MaximumStreamOrder-2)]
MA_VS_Max2['ZoneId'] = 'MA_VS_Max2'
Zones.append(MA_VS_Max2) # add index to list

# ----------- High ALTITUDE -------------------
HighAltitude = All_static_data.loc[(All_static_data['elevation'] > 800)]

# ----------- Gentle gradient -------------------
HA_GentleGradient = HighAltitude.loc[(HighAltitude['slope'] <= 0.02)]

HA_GG_Max = HA_GentleGradient.loc[(HA_GentleGradient['stream_order'] == MaximumStreamOrder)]
HA_GG_Max['ZoneId'] = 'HA_GG_Max'
Zones.append(HA_GG_Max) # add index to list
HA_GG_Max1 = HA_GentleGradient.loc[(HA_GentleGradient['stream_order'] == MaximumStreamOrder-1)]
HA_GG_Max1['ZoneId'] = 'HA_GG_Max1'
Zones.append(HA_GG_Max1) # add index to list
HA_GG_Max2 = HA_GentleGradient.loc[(HA_GentleGradient['stream_order'] == MaximumStreamOrder-2)]
HA_GG_Max2['ZoneId'] = 'HA_GG_Max2'
Zones.append(HA_GG_Max2) # add index to list

#------------- riffles -----------------------
HA_riffle = HighAltitude.loc[(HighAltitude['slope'] > 0.02) & (HighAltitude['slope'] <= 0.04)]

HA_RI_Max = HA_riffle.loc[(HA_riffle['stream_order'] == MaximumStreamOrder)]
HA_RI_Max['ZoneId'] = 'HA_RI_Max'
Zones.append(HA_RI_Max) # add index to list
HA_RI_Max1 = HA_riffle.loc[(HA_riffle['stream_order'] == MaximumStreamOrder-1)]
HA_RI_Max1['ZoneId'] = 'HA_RI_Max1'
Zones.append(HA_RI_Max1) # add index to list
HA_RI_Max2 = HA_riffle.loc[(HA_riffle['stream_order'] == MaximumStreamOrder-2)]
HA_RI_Max2['ZoneId'] = 'HA_RI_Max2'
Zones.append(HA_RI_Max2) # add index to list

# ----------- steep -----------------------
HA_steep = HighAltitude.loc[(HighAltitude['slope'] > 0.04) & (HighAltitude['slope'] <= 0.1)]

HA_ST_Max = HA_steep.loc[(HA_steep['stream_order'] == MaximumStreamOrder)]
HA_ST_Max['ZoneId'] = 'HA_ST_Max'
Zones.append(HA_ST_Max) # add index to list
HA_ST_Max1 = HA_steep.loc[(HA_steep['stream_order'] == MaximumStreamOrder-1)]
HA_ST_Max1['ZoneId'] = 'HA_ST_Max1'
Zones.append(HA_ST_Max1) # add index to list
HA_ST_Max2 = HA_steep.loc[(HA_steep['stream_order'] == MaximumStreamOrder-2)]
HA_ST_Max2['ZoneId'] = 'HA_ST_Max2'
Zones.append(HA_ST_Max2) # add index to list

# --------------- very steep ----------------
HA_verysteep = HighAltitude.loc[(HighAltitude['slope'] > 0.1)]

HA_VS_Max = HA_verysteep.loc[(HA_verysteep['stream_order'] == MaximumStreamOrder)]
HA_VS_Max['ZoneId'] = 'HA_VS_Max'
Zones.append(HA_VS_Max) # add index to list
HA_VS_Max1 = HA_verysteep.loc[(HA_verysteep['stream_order'] == MaximumStreamOrder-1)]
HA_VS_Max1['ZoneId'] = 'HA_VS_Max1'
Zones.append(HA_VS_Max1) # add index to list
HA_VS_Max2 = HA_verysteep.loc[(HA_verysteep['stream_order'] == MaximumStreamOrder-2)]
HA_VS_Max2['ZoneId'] = 'HA_VS_Max2'
Zones.append(HA_VS_Max2) # add index to list

# convert to dataframe
ZonesData = pd.concat(Zones, axis = 0, ignore_index = True)    # convert to dataframe
    
nameCSV = 'Zones' + '.csv'
AllDataPath = os.path.join(output, nameCSV)  
ZonesData.to_csv(AllDataPath, sep=';', mode="w", header=True, index=False)   


#%% calculate simple discharge statistics (5P, median, 95P)

qr_median = np.median(qr, axis = 0)
qr_5p = np.percentile(qr,5, axis =0)
qr_95p = np.percentile(qr,95, axis =0)

# add discharge stats to coordinates
data_step = pd.DataFrame(data=qr_median)
# turn dataframe upside down to match the static data..?
data_step = pd.DataFrame(data=data_step).iloc[::-1]
medianQ = []
for i in range(len(long)):
    df3 = pd.DataFrame(data_step[i])
    frames = [df3]
    data = pd.concat(frames, axis = 1, ignore_index = True)    
    data.reset_index(drop=True, inplace=True)
    medianQ.append(data)
medianQ_complete = pd.concat(medianQ, axis = 0, ignore_index = True)   


data_step = pd.DataFrame(data=qr_5p)
# turn dataframe upside down to match the static data..?
data_step = pd.DataFrame(data=data_step).iloc[::-1]
qr_5p = []
for i in range(len(long)):
    df3 = pd.DataFrame(data_step[i])
    frames = [df3]
    data = pd.concat(frames, axis = 1, ignore_index = True)    
    data.reset_index(drop=True, inplace=True)
    qr_5p.append(data)
qr_5p_complete = pd.concat(qr_5p, axis = 0, ignore_index = True)   

data_step = pd.DataFrame(data=qr_95p)
# turn dataframe upside down to match the static data..?
data_step = pd.DataFrame(data=data_step).iloc[::-1]
qr_95p = []
for i in range(len(long)):
    df3 = pd.DataFrame(data_step[i])
    frames = [df3]
    data = pd.concat(frames, axis = 1, ignore_index = True)    
    data.reset_index(drop=True, inplace=True)
    qr_95p.append(data)
qr_95p_complete = pd.concat(qr_95p, axis = 0, ignore_index = True)   

