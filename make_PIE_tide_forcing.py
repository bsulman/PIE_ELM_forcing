import xarray
import pandas
import numpy as np

# In order from most to least saline, Railroad, Middle Road, Typha
RR_data=pandas.read_csv('RR_tide_salinity_1423_gf_1002.csv',parse_dates=['timestamp'])
MR_data=pandas.read_csv('MR_tide_salinity_1423_gf_1002.csv',parse_dates=['timestamp'])
Typha_data=pandas.read_csv('Typha_tide_salinity_1423_gf_1002.csv',parse_dates=['timestamp'])

# Saltwater barrier: https://waterservices.usgs.gov/nwis/iv/?format=rdb&sites=08041780&startDT=2008-01-01T00:00&endDT=2023-12-31T23:59&siteStatus=all
# Green Pond Gully: 08041940

# Tide heights in NAVD88. When using with ELM, make sure humhol_ht parameter is correct for marsh soil level relative to NAVD88
tide_height=xarray.Variable(('time','gridcell'),
            data=np.column_stack((RR_data['tide_NAVD88'],MR_data['tide_NAVD88'],Typha_data['tide_NAVD88'])),
            attrs={'units':'m'})
tide_salinity=xarray.Variable(('time','gridcell'),
            data=np.column_stack((RR_data['Salinity_ppt'],MR_data['Salinity_ppt'],Typha_data['Salinity_ppt'])),
            attrs={'units':'m'})

time=xarray.Variable(('time',),data=MR_data['timestamp'],attrs={})

# Putting the data into an xarray data structure and resampling to one hour time step to match ELM time step
tide_data=xarray.Dataset(
    data_vars={'tide_height':tide_height,'tide_salinity':tide_salinity.clip(min=1e-5)},
    coords   ={'time':time,'gridcell':[0,1,2]},
    attrs    ={'Description':'Tide data for Railroad, Middle Road, and Typha sites of Plum Island LTER. Data provided by Inke Forbrich'}
).resample(time='1h').mean().interpolate_na(dim='time')

# Add nitrate field to forcing data. ELM-PFLOTRAN requires nitrate, but we don't have data so using a constant value
tide_data['tide_nitrate']=xarray.Variable(('time','gridcell'),data=np.zeros((len(tide_data['time']),3))+0.3e-3,attrs={'units':'mol/L'})

# This netCDF file will have tide level, salinity, and nitrate concentration time series for the three sites (so it's set up for a three grid cell simulation of the three sites simultaneously)
tide_data.to_netcdf('PIE_tide_forcing.nc')

# Talking with Inke Aug 20: 2022 and 2023 had the biggest difference and easier to link to observations. So let's swap those. 2022 is the outlier.
# Should also rerun both swapped and control to make sure timing is right.
# Do we see a difference in BGC between high and low marsh due to different inundation patterns (i.e. antecedent condition)?
# Is there a difference in how fast water table draws down due to different VPD across years
tide_data_swapyears=tide_data.copy(deep=True)
tide_data_swapyears['tide_height'][tide_data_swapyears['time'].to_index().year==2023]=tide_data['tide_height'][tide_data['time'].to_index().year==2022].values
tide_data_swapyears['tide_height'][tide_data_swapyears['time'].to_index().year==2022]=tide_data['tide_height'][tide_data['time'].to_index().year==2023].values
tide_data_swapyears['tide_salinity'][tide_data_swapyears['time'].to_index().year==2023]=tide_data['tide_salinity'][tide_data['time'].to_index().year==2022].values
tide_data_swapyears['tide_salinity'][tide_data_swapyears['time'].to_index().year==2022]=tide_data['tide_salinity'][tide_data['time'].to_index().year==2023].values
tide_data_swapyears.to_netcdf('PIE_tide_forcing_swapyears.nc')

tide_data_2323=tide_data.copy(deep=True)
#tide_data_2323['tide_height'][tide_data_swapyears['time'].to_index().year==2023]=tide_data['tide_height'][tide_data['time'].to_index().year==2022].values
tide_data_2323['tide_height'][tide_data_2323['time'].to_index().year==2022]=tide_data['tide_height'][tide_data['time'].to_index().year==2023].values
#tide_data_2323['tide_salinity'][tide_data_swapyears['time'].to_index().year==2023]=tide_data['tide_salinity'][tide_data['time'].to_index().year==2022].values
tide_data_2323['tide_salinity'][tide_data_2323['time'].to_index().year==2022]=tide_data['tide_salinity'][tide_data['time'].to_index().year==2023].values
tide_data_2323.to_netcdf('PIE_tide_forcing_2323.nc')

# Next we will set up the domain and surface files for this three grid cell configuration.
# We start by opening the single column configuration files from the original one cell model simulation
domain_onecol=xarray.open_dataset('domain_PIE_onecol.nc')
surfdata_onecol=xarray.open_dataset('surfdata_PIE_onecol.nc')

# Set up three cell domain file by concatenating the original one
num_grids=3
domain_multicell=xarray.concat([domain_onecol]*num_grids,dim='ni')
# Assume all grid cells are the same size, just dividing up the size of the original domain cell.
cell_width=(domain_onecol['xv'].max()-domain_onecol['xv'].min()).item()/num_grids
domain_multicell['xc'][0,:] = domain_multicell['xc'][0,0].item()+np.arange(num_grids)*cell_width
domain_multicell['xv'][0,:,[0,2]] =  domain_multicell['xc'].to_numpy().T + cell_width/2
domain_multicell['xv'][0,:,[1,3]] =  domain_multicell['xc'].to_numpy().T - cell_width/2
# Currently leaving yv alone, which means cells are not square exactly
domain_multicell['area'] = domain_multicell['area']/num_grids

domain_multicell.to_netcdf('PIE_domain_threecell.nc')

# Same approach for surface properties data file
# Expand surface data along grid cell axis to match domain
surfdata_multicell = xarray.concat([surfdata_onecol]*num_grids,dim='lsmlon',data_vars='minimal')
surfdata_multicell['LONGXY']=domain_multicell['xc']

# Add new fields specific to gridded tidal forcing
# ht_above_stream is the level of the marsh surface in each site, in the same datum as the tide levels (i.e., NAVD88) so make sure that's correct
surfdata_multicell['ht_above_stream'] = xarray.DataArray(data=np.zeros((1,num_grids)),dims={'lsmlat':1,'lsmlon':num_grids}) + 0.252

# dist_from_stream is lateral distance from the edge of the marsh to where the site is defined, which affects how tightly coupled 
# the tides in the model column are to the tide forcing. So, a short distance for the creek bank or a longer distance for the marsh interior
surfdata_multicell['dist_from_stream'] = xarray.DataArray(data=np.zeros((1,num_grids)),dims={'lsmlat':1,'lsmlon':num_grids}) + 4.0

# PCT_NAT_PFT controls what plant functional type is used for the grid cell. 
# The current configuration used PFT 14 (C4 grass), but we can change that if we want to represent different veg types
# The model is also sensitive to organic matter concentration (ORGANIC in the surface data file) which is defined for each depth
# layer in each grid cell. ORGANIC is in units of kg/m3, and is assumed to be relative to a maximum organic matter concentration of 130 kg/m3

surfdata_multicell['PCT_NAT_PFT'][13,:,2]=100.0
surfdata_multicell['PCT_NAT_PFT'][14,:,2]=0.0

surfdata_multicell.to_netcdf('PIE_surfdata_threecell.nc')
