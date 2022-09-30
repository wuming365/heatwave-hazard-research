# Mapping of Study Area

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import pandas as pd
from osgeo import gdal
from matplotlib import cm
import os
from tqdm import trange
import tqdm

def mapping(language='en'):
    
    def init_background(mapn, axn,dem_path=None,shape_path=None):
        
        # these strings are all excluding extension filename, e.g. ".shp"
        china_path = r"" # china polygon file
        world_path = r"" # world countrys polygon file
        line_path = r"" # boundry of China line file 
        province_path = r"" # provincial boundaries (excluding national borders)
        geograpically_path = r"" # divisions polygon file
        
        # to show countries other than China
        mapn.readshapefile(world_path,
                           name='world',
                           drawbounds=False,
                           default_encoding='gbk')
        df = pd.DataFrame(mapn.world_info)
        not_china_index = np.array(df[df["NAME"] != "CHINA"].index)
        del df
        p1 = PatchCollection(
            [
                Polygon(i)
                for i in np.array(mapn.world, dtype=object)[not_china_index]
            ],
            edgecolors='black',
            facecolors='lightgray',
            linewidths=0.2,
        )
        axn.add_collection(p1)
        del p1
        
        # to show China polygon file
        mapn.readshapefile(china_path, name='china')
        # matplotlib.collections.PatchCollection can fill the polygon with setting color, but readshapefile() can only set boundary of polygon, which will create gaps           where there is no data. e.g. Taiwan, Hongkong
        p2 = PatchCollection(
            [Polygon(i) for i in np.array(mapn.china, dtype=object)],
            edgecolors='black',
            facecolors='lightgray',
            linewidth=0.8)
        axn.add_collection(p2)
        del p2
        
        #to display the boundary of China
        mapn.readshapefile(line_path, name='china_line', linewidth=0.8)

        # to display dem data, I haven't display dem in accompanying drawing because this data is so large that will slow down the charting speed and is not very                 necessary
        if dem_path is not None:
            ds=gdal.Open(dem_path)
            data = ds.ReadAsArray()
            (xmin, xl, _, ymax, _, _) = ds.GetGeoTransform()
            width = data.shape[1]
            height = data.shape[0]
            ndv = ds.GetRasterBand(1).GetNoDataValue()
            data = np.ma.masked_where(data == ndv, data)
            ymin = ymax - height * xl
            xmax = xmin + width * xl
            x = np.linspace(xmin, xmax, width)
            y = np.linspace(ymin, ymax, height)[::-1]
            xx, yy = np.meshgrid(x, y)
            ticks = [-1000,0, 200, 500, 1000,1500, 2000, 3000, 4000, 5000, 6000,9000]
            colors=plt.cm.get_cmap("gist_earth")(np.linspace(0.5, 1,11))
            cntr = mapn.contourf(xx, yy, data,levels=ticks,colors=colors)
            colorbar = mapn.colorbar(cntr, size='3.5%', ticks=ticks[1:-1])
            colorbar.ax.tick_params(labelsize=14)
            
        # to display the weather stations
        if shape_path is not None:
            df = pd.read_csv(shape_path)
            # to get the latitude and longitude of training set and vadidation set to display them differently
            chazhi_lon = np.array(df[df["type"] != "training"]["longitude"])
            chazhi_lat=np.array(df[df["type"] != "training"]["latitude"])
            yanzheng_lon = np.array(df[df["type"] != "validation"]["longitude"])
            yanzheng_lat = np.array(df[df["type"] != "validation"]["latitude"])
            del df
            chazhi=mapn.scatter(chazhi_lon, chazhi_lat,c="red",s=2)
            yanzheng = mapn.scatter(yanzheng_lon, yanzheng_lat, c="blue",s=2)
            axn.legend(handles=[chazhi, yanzheng],
                       loc='lower left',
                       labels=[
                           "Monitoring Station:Training Set, 450",
                           "Monitoring Station:Validation Set, 241"
                       ],
                       fontsize=15)
        
        # to display the provincal boundary data
        mapn.readshapefile(
            province_path,
            name='province',
            linewidth=0.2,
        )
        
        #to display the geograpical division boundary data
        mapn.readshapefile(
            geograpically_path,
            name='geograpically',
            linewidth=0.5,
        )
    
    #to set the normal font and display "-"
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(12, 9), dpi=200,facecolor='white')
    left, bottom, width, height = 0.025, 0.05, 0.95, 0.95
    ax1 = fig.add_axes([left, bottom, width, height])
    
    #to create the main image
    map1 = Basemap(llcrnrlon=73,
                    llcrnrlat=18,
                    urcrnrlon=136,
                    urcrnrlat=54,
                    ax=ax1)
    
    # full filename
    dem_path = r"" # dem raster data
    stations_path = r"" # a csv file including longitude , latitude, id, type
    
    # draw the main image
    init_background(map1, ax1, dem_path, stations_path)
    # draw the parallels in the main image
    map1.drawparallels(np.linspace(20, 50, 7),
                        labels=[1, 0, 0, 0],
                        linewidth=0,
                        fontsize=14)
    map1.drawmeridians(np.linspace(75, 135, 7),
                       labels=[0, 0, 0, 1],
                       linewidth=0,
                       fontsize=14)
    
    # to create accompanying drawing
    ax2 = fig.add_axes([0.775, 0.15, 0.15, 0.3])
    map2 = Basemap(llcrnrlon=106,
                    llcrnrlat=2,
                    urcrnrlon=124,
                    urcrnrlat=23,
                    ax=ax2)
    init_background(map2, ax2)
    # to add the text (if need)
    ax2.text(
            0.5,
            0.02,
            horizontalalignment='center',
            verticalalignment='bottom',
            s=f"South China Sea",
            transform=ax2.transAxes,
            fontsize=15,
        )
    
    # to add the title of legend of colorbar in the figure
    fig.text(0.93, 0.88, 'Elevation(m)', fontsize=12)
    
    plt.show() # or can directly save fig.savefig()
    
if __name__=="__main__":
    mapping()
