import xarray as xr 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cftime
import scipy.stats as stats
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

# colormap for zonal rh plots
jet_cmap = plt.colormaps.get_cmap("jet") 
jet_cmap.set_under("midnightblue") 
jet_cmap

def zonal_mean_rh_plot_for_grid(ds, ax, exp = '', zoom = 0, levels = 100, negative= False , cmap1=jet_cmap, rh_cmap_label= 'Relative Humidity (%)', label_contours=True, contour_levels = [50, 60, 70, 80, 90, 100], colorbar=False, seasonal_avg=False, Season = '', sub_label_season=True, extend = 'min'):
    '''Zoom default is 0, then the latitude ranges from -90 to +90, 
    but if zoom is nonzero, the x axis range is +- the value of zoom (e.g. zoom =25) degrees'''
    '''The way it returns cf means it stores the values to make the colorbar when you plot several in a grid'''
    '''Default is to plot yearly average but if seasonal_avg is true then this plots a seasonal average for the selected Season'''
    if seasonal_avg:
        mean = ds.groupby('time.season').mean(dim = ['time','lon']).sel(season=Season).rh # seasonal average over specified season
    else:
        mean = ds.rh.mean(dim = ['time','lon']) # Yearly averaged

    
    EXTEND = extend
        
    cf = ax.contourf(mean.lat, mean.pfull, mean.values, extend=EXTEND, cmap=cmap1, levels=levels, zorder=10)
        
    if label_contours:
        contours = ax.contour(mean.lat, mean.pfull, mean.values, contour_levels, extend='neither', zorder=20, colors='black')
        labels = ax.clabel(contours, inline= True, colors='black',fontsize=10.5, zorder=12)
    
    if zoom != 0:
        ax.set_xlim(-zoom, zoom)
        
    ax.invert_yaxis()  
    ax.set_xlabel('Latitude ($^\circ$N)', fontsize =13)
    ax.set_ylabel('Pressure (hPa)', fontsize =13)
    if sub_label_season:
        ax.set_title(f'{exp} {Season}', fontsize =14)
    else:
        ax.set_title(f'{exp}', fontsize = 14)
    return cf

def colorbar(c, figure, colorbar_label = '', colorbar_ticks = None):
    '''Add a colorbar as a subplot, where c is the contour set you want the colorbar to reflect'''

    # Adjust the location of the subplots on the page to make room for the colorbar
    figure.subplots_adjust(bottom=0.1, top=0.95, left=0.001, right=0.99, wspace=0.1, hspace=0.5)

    # Add a colorbar axis at the bottom of the graph
    cbar_ax = figure.add_axes([1, 0.2, 0.02, 0.6])  # left, bottom, width, height

    # Draw the colorbar
    cbar=figure.colorbar(c, ticks=colorbar_ticks, cax=cbar_ax)
    cbar.set_label(colorbar_label ,fontsize=13)
    return cbar

def precipitation_plot(data, labels, zoom = 0, exp = '', colours = None, styles = None, seasonal_avg=False, Season = '', ymin = 0, ymax = 500):
    for i in range(len(data)):
        ds = data[i]
        colour = colours[i]
        style = styles[i]
        if seasonal_avg:
            mean = ds.groupby('time.season').mean(dim = ['time','lon']).sel(season=Season).precipitation # seasonal average over specified season
            fig = (mean*2.26e6).plot(linestyle = style, color = colour, label = labels[i])
        else:
            fig = (ds.precipitation.mean(dim=['time', 'lon'])*2.26e6).plot(linestyle = style ,color = colour, label = labels[i])
            
        plt.legend()
        if zoom != 0:
            plt.xlim(-zoom, zoom)
        plt.ylim(ymin, ymax)
    plt.xlabel('Latitude ($^\circ$N)')
    plt.ylabel('Precipitation (W/$m^2$)')
    #plt.title(f'{exp}' +'\n Zonal Mean Precipitation')
    return fig

def precipitation_at_equator(ds):
    max_precip =  (ds.precipitation.mean(dim = ['time', 'lon'])*2.26e6).sel(lat=0, method = 'nearest').values
    return max_precip

def max_precip(ds):
    max_precip =  (ds.precipitation.mean(dim = ['time', 'lon'])*2.26e6).values.max()
    return max_precip

def cumulative_mean_precip(ds, seasonal_avg = False, latmin = -30, latmax = 30, Season = None):
    
    ds = ds.sel(lat = slice(latmin, latmax)) # limit to latmin degrees south of equator, latmax degrees north (default tropics)
    if seasonal_avg:
        seasonal = ds.groupby('time.season').mean(dim = ['time','lon']).sel(season=Season).precipitation
        total_precip = (seasonal*2.26e6).values.sum()
    else:
        total_precip = (ds.mean(dim = ['time', 'lon']).precipitation*2.26e6).values.sum()
    
    return total_precip

def precipitation_max(ds, seasonal_avg = False, latmin = -30, latmax = 30, Season = None):
    
    ds = ds.sel(lat = slice(latmin, latmax)) # limit to latmin degrees south of equator, latmax degrees north (default tropics)
    if seasonal_avg:
        seasonal = ds.groupby('time.season').mean(dim = ['time','lon']).sel(season=Season).precipitation
        max_precip_tropics = (seasonal*2.26e6).values.max()
    else:
        max_precip_tropics = (ds.mean(dim = ['time', 'lon']).precipitation*2.26e6).values.max()
    
    return max_precip_tropics

def global_mean_temp_2m(ds, plotting=True, legendlabel=None):
    weights=np.cos(np.deg2rad(ds.lat))
    weighted_temp = ds.temp_2m.weighted(weights)
    weighted_mean_temp = weighted_temp.mean(("lon", "lat"))
    if plotting == True:
        weighted_mean_temp.plot(label= legendlabel)
    return weighted_mean_temp

def global_mean_precipitation_mmpmonth(ds, p=1000, plotting=True, legendlabel=None):
    weights=np.cos(np.deg2rad(ds.lat))
    weighted_precip = ds.precipitation.weighted(weights)
    weighted_mean_precip = (weighted_precip.mean(("lon", "lat")))
    weighted_mean_precip_mmpermonth = weighted_mean_precip*8640*30
    if plotting == True:
        weighted_mean_precip_mmpermonth.plot(label= legendlabel)
    return weighted_mean_precip_mmpermonth

def global_mean_temperature(ds, p=1000, plotting=True):
    pressure_surface_data=ds.sel(pfull=p, method='nearest') #default earth surface p=1000hpa
    weights=np.cos(np.deg2rad(pressure_surface_data.lat))
    weighted_temp = pressure_surface_data.temp.weighted(weights)
    weighted_mean_temp = weighted_temp.mean(("lon", "lat"))
    if plotting == True:
        weighted_mean_temp.plot()
    return weighted_mean_temp

def global_mean_pert_t(ds, p=1000, plotting=True):
    pressure_surface_data=ds.sel(pfull=p, method='nearest') #default earth surface p=1000hpa
    weights=np.cos(np.deg2rad(pressure_surface_data.lat))
    weighted_temp = pressure_surface_data.pert_t.weighted(weights)
    weighted_mean_temp = weighted_temp.mean(("lon", "lat"))
    if plotting == True:
        weighted_mean_temp.plot()
    return weighted_mean_temp

def spatial_plot(ds, variable, ax, p = 1000, C = 'plasma',  cbar_label = None, c_levels = 60):
    ds = ds.sel(pfull = p, method = 'nearest')
    var = ds[variable].mean('time')
    lon = ds['lon']
    lat = ds['lat']
    # Add a cyclic point to the data and longitude
    var_cyclic, lon_cyclic = add_cyclic_point(var, coord=lon)

    ax.set_global()
    ax.coastlines()

    # Plot using contourf
    contour = ax.contourf(lon_cyclic, lat, var_cyclic, levels = c_levels , cmap = C, extend='both', transform=ccrs.PlateCarree())
    
    # Add a colorbar
    if cbar_label:
        plt.colorbar(contour, ax=ax, label=cbar_label, shrink=0.8)
    else:
        plt.colorbar(contour, ax=ax, label=variable, shrink=0.8)

    return ax;

def spatial_plot_gridded(ds1, ds2, variable, pressure = 1000, cbar_labels = [None, None, 'difference'], subplot_labels = ['(a) ML', '(b) Control', '(c) ML - Control'] , contour_levels = [60, 60, 60], cmap = 'viridis'):
    '''Default labels assume ds1 is the ML exp, ds2 the control, hence the difference is ds1 - ds2 = ML - control'''
    t_statistic, p_value = stats.ttest_ind(ds1[variable].sel(pfull = pressure, method = 'nearest'), ds2[variable].sel(pfull = pressure, method = 'nearest'), equal_var = False)
    lon = ds1['lon']
    lat = ds1['lat']
    fig, axs = plt.subplots(ncols = 1, nrows = 3, figsize=(8, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    spatial_plot(ds1, variable, ax = axs[0],p = pressure, cbar_label = cbar_labels[0], c_levels = contour_levels[0], C = cmap);
    spatial_plot(ds2, variable, ax = axs[1],p = pressure, cbar_label = cbar_labels[1], c_levels = contour_levels[1], C = cmap);
    spatial_plot(ds1 - ds2, variable, ax = axs[2],p = pressure, cbar_label = cbar_labels[2], c_levels = contour_levels[2], C = 'seismic');
    axs[2].contour(lon, lat, p_value, levels=[0.05], colors='black', transform=ccrs.PlateCarree())
    axs[2].contourf(lon, lat, p_value, levels=[0, 0.05], colors='none', hatches=['xxxx', ''], transform=ccrs.PlateCarree())
    for i in [0, 1, 2]:
        axs[i].set_title(subplot_labels[i])
    fig.tight_layout()
    return fig

def spatial_plot_noverticaldim(ds, variable, ax, C = 'viridis',  cbar_label = None, c_levels = 60):
    var = ds[variable].mean('time')
    lon = ds['lon']
    lat = ds['lat']

    if variable == 'precipitation':
        var = var*2.26e6 # convert precipitation to w/m2
    # Add a cyclic point to the data and longitude
    var_cyclic, lon_cyclic = add_cyclic_point(var, coord=lon)

    ax.set_global()
    ax.coastlines()

    # Plot using contourf
    
    contour = ax.contourf(lon_cyclic, lat, var_cyclic, levels = c_levels , cmap = C, extend='both', transform=ccrs.PlateCarree())
    
    # Add a colorbar
    if cbar_label:
        plt.colorbar(contour, ax=ax, label=cbar_label, shrink=0.8)
    else:
        plt.colorbar(contour, ax=ax, label=variable, shrink=0.8)

    return ax;

def spatial_plot_gridded_noverticaldim(ds1, ds2, variable, cbar_labels = [None, None, 'difference'], subplot_labels = ['(a) ML', '(b) Control', '(c) ML - Control'], contour_levels = [60, 60, 60], cmap = 'viridis'):
    '''Default labels assume ds1 is the ML exp, ds2 the control, hence the difference is ds1 - ds2 = ML - control'''
    t_statistic, p_value = stats.ttest_ind(ds1[variable], ds2[variable], equal_var = False)
    fig, axs = plt.subplots(ncols = 1, nrows = 3, figsize=(10, 14), subplot_kw={'projection': ccrs.PlateCarree()})
    spatial_plot_noverticaldim(ds1, variable, ax = axs[0], cbar_label = cbar_labels[0], c_levels = contour_levels[0], C = cmap);
    spatial_plot_noverticaldim(ds2, variable, ax = axs[1], cbar_label = cbar_labels[1], c_levels = contour_levels[1], C = cmap);
    spatial_plot_noverticaldim(ds1 - ds2, variable, ax = axs[2], cbar_label = cbar_labels[2], c_levels = contour_levels[2], C = 'seismic');

    lon = ds1['lon']
    lat = ds1['lat']
    # Add a cyclic point to the data and longitude
    p_cyclic, lon_cyclic = add_cyclic_point(p_value, coord=lon)
    
    axs[2].contour(lon_cyclic, lat, p_cyclic, levels=[0.05], colors='black', transform=ccrs.PlateCarree())
    axs[2].contourf(lon_cyclic, lat, p_cyclic, levels=[0, 0.05], colors='none', hatches=['xxxx', ''], transform=ccrs.PlateCarree())
    for i in [0, 1, 2]:
        axs[i].set_title(subplot_labels[i])
    fig.tight_layout()
    return fig

def plotting_gm_timeseries(data,var, ylabel = None, legendlabel=None):
    ''' Plots a global mean timeseries for an datasets
    
    Parameters:
        data: list of xarray datasets
        var: str, variable name to plot
        ylabel: str, label for the y-axis (optional)
        legendlabel: list of str, labels for the legend corresponding to datasets (optional)''' 
    days = np.unique([date.day for date in data[0].time.values])
    years = np.unique([date.year for date in data[0].time.values])
    dates = [cftime.Datetime360Day(year, month, day) for year in years for month in range(1, 13) for day in days] 
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(data)):
        ds = data[i]
        weights = np.cos(np.deg2rad(ds.lat))
        weighted_var = ds[var].weighted(weights)
        weighted_mean_var = weighted_var.mean(("lon", "lat"))
        if var == 'precipitation':
            weighted_mean_var = weighted_mean_var*60*60*24*30 # convert from kg/m2/s = mm/s to mm/month (1month = 60s x 60min x 24h x 30days)
        ax.plot(dates, weighted_mean_var.values, label = legendlabel[i], linestyle = '-', linewidth = 1)  
    # Adding the next year to the array of years so that the x labels includes the next year
    years = np.append(years, years[-1] + 1)
    ax.set_xticks([cftime.Datetime360Day(year, 1, days[0]) for year in years])
    ax.set_xticklabels(years - years[0], fontsize = 10)
    ''' The above is to label the x axis with the number of years since the start of the experiment run'''
    if ylabel:
        plt.ylabel(ylabel)
    else:
        plt.ylabel(var) # default label is the variable name, unless you set an alternative ylabel
    plt.xlabel('Time (years)')
    plt.title(f'Global Mean {var} ML Experiments')
    plt.legend(bbox_to_anchor=(1, 1))
    return fig







