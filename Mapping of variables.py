# Mapping of variables

# Attention: Figure subfigures need a higher version

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
import matplotlib.colors as colors

# to truncate the colormap to needed colormap
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap

def mapping(img_path,
            title_keys=None,
            title=None,
            statistics=False,
            language='en',
            ):
    def init_background(mapn,
                        axn,
                        sfig,
                        unit,
                        key,
                        data_path=None,
                        statistics=False):
        
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
        
        if data_path is not None: # if there is a file
            ds = gdal.Open(data_path)
            data = ds.ReadAsArray()
            (xmin, xl, _, ymax, _, _) = ds.GetGeoTransform()
            width = data.shape[1]
            height = data.shape[0]
            ndv = ds.GetRasterBand(1).GetNoDataValue()
            
            # if this is a "slope" file of ordinary variables (not slope_hazard), we need to calculate the proportion of areas greater than 0
            if "slope" in data_path and "hazard" not in data_path:
                data_b = data[data != ndv]
                data_c = data_b[data_b >= 0]

                percent = len(data_c) / len(data_b)
                print(data_path, percent)
               
            data = np.ma.masked_where(data == ndv, data) # mask the ndv or use ma.masked_equal()
            data_min, data_max = np.nanmin(data), np.nanmax(data)
            
            # if we need to statistic the raster data
            if statistics:
                data1 = data[data != np.ma.masked]
                total_num = 9576022
                result = {
                    # 'year': int(data_path.split("\\")[-1][:4]),
                    'name': data_path.split("\\")[-1][:-4],
                    'min': data_min,
                    'max': data_max,
                    'per_min': len(data1[data1 == data_min]) / total_num,
                    'per_max': len(data1[data1 == data_max]) / total_num,
                    'min_per_1': np.average(np.sort(data1)[:2000]),
                    'max_per_1': np.average(np.sort(data1)[-2020:]),
                    'average': np.average(data1),
                }
                del data1
                result = pd.DataFrame(result, index=[0])
            
            # read the 2-d meshgrid    
            ymin = ymax - height * xl
            xmax = xmin + width * xl
            x = np.linspace(xmin, xmax, width)
            y = np.linspace(ymin, ymax, height)[::-1]
            xx, yy = np.meshgrid(x, y)
            
            # if display our avg_hazard and slope_hazard data
            if "hazard" in data_path and "Yin" not in data_path:
                cntr = mapn.contourf(xx, yy, data, levels=5, cmap='RdYlBu_r')
                ticks=[0.4,1.2,2.0,2.8,3.6]
            # if display the ordinary variables and slope of them
            else:
                ticks = np.sort(
                list(set((np.linspace(data_min, data_max, 80) /
                        unit).astype(int)))) * unit
                if "slope" in data_path:
                    # to make sure that red is greater than 0 and blue is less than 0
                    if data_min>=0:
                        cmap = truncate_colormap(mpl.cm.get_cmap("RdYlBu_r"),0.5,1)
                    elif data_max<0:
                        cmap = truncate_colormap(mpl.cm.get_cmap("RdYlBu_r"),0,0.5)
                    else:
                        cmap = mpl.cm.get_cmap("RdYlBu_r")
                    cntr = mapn.contourf(xx, yy, data,levels=80, cmap=cmap)
                else:
                    cntr = mapn.contourf(xx, yy, data, levels=80, cmap='RdYlBu_r')
            
            # delete the ticks out of range
            if ticks[0] < data_min:
                    ticks = np.delete(ticks, 0)
            if ticks[-1] > data_max:
                ticks = np.delete(ticks, -1)
        
        # create the colorbar
        colorbar = mapn.colorbar(cntr, size='3.5%', ticks=ticks,fig=sfig)
        colorbar.ax.tick_params(labelsize='21')
        
        # reset the ticks of colorbar
        if "hazard" in key and "Yin" not in key:
            newticks=["Low","Medium low","Middle","Medium high","High"]
            if "slope" in key:
                newticks=["Decreased", "Slightly decreased", "Basically unchanged","Slightly increased", "Increased"]
            colorbar.set_ticklabels(newticks)
            
        del data
        del ds
        
        mapn.readshapefile(
                province_path,
                name='province',
                linewidth=0.2,
            )
        
        # if display slope_hazard, then load the p value shape
        if "hazard_slope" in key:
                p_value_path = r"F:\dem_chazhi\result\hazard\p_value.txt"
                p_value_polyline = r"F:\dem_chazhi\result\hazard\p_value"
                df = pd.read_csv(p_value_path)
                lonchazhi = np.array(df["x"][:])
                latchazhi = np.array(df["y"][:])
                mapn.readshapefile(p_value_polyline,
                                   name='p_value',
                                   linewidth=0.4)
                mapn.scatter(lonchazhi,
                             latchazhi,
                             c='b',
                             s=1,
                             alpha=0,
                             label="P<0.05")
                ax1.legend(loc='lower left', prop={'size': 30})
            
            if statistics:
                return result
            
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    
    results = pd.DataFrame()
    letter = list(map(chr, range(ord('a'), ord('z') + 1))) # add the identity of every sub-figures
    
    # size of one sub-figure is (12.5, 6.67)
    nums = len(title_keys)
    rows = int(nums/2) + 1
    cols = nums>2?2:1
    
    fig = plt.figure(
    figsize=(rows * 12.5, cols * 10 / 3), 
    dpi=300,
    facecolor="white",
    )
    if cols==1:
        sfigs=np.array([fig.subfigures(1,1)])
    else:
        sfigs=fig.subfigures(rows,2)
    fig.subplots_adjust(left=0,
                        bottom=0,
                        right=1,
                        top=0.9,
                        wspace=0.05,
                        hspace=0.05)
    
    # Traversing and mapping each sub-figures
    for i in trange(nums):
        data_path = os.path.join(img_path, title_keys[i] + ".tif")
        left, bottom, width, height = 0.025, 0.05, 0.95, 0.95
        
        # to calculate the coordinate of each sub-figures
        if len(sfigs.shape)==1:
            sfig = sfigs[i]
        else:
            sfig = sfigs[int(i/2)][i%2]
        
        # add identity only if multi-sub-figures
        if nums!=1:
            sfig.text(
                0.09,
                0.97,
                horizontalalignment='center',
                verticalalignment='top',
                s=letter[i],
                # transform=sfig.transAxes,
                fontsize=45,
                fontweight='medium')
        
        # add main image for sub-figure
        ax1 = sfig.add_axes([0.055, 0.07, 0.9, 0.95])
        map1 = Basemap(llcrnrlon=73,
                       llcrnrlat=18,
                       urcrnrlon=136,
                       urcrnrlat=54,
                       ax=ax1)
        key=title_keys[i]
        
        # if this is average of 30 years
        if "_1990" in key:
            unit = title[title_keys[i][:-10]][2]
        else:
            unit = title[title_keys[i]][2]
        
        # if need statistic
        if statistics:
            result = init_background(
                map1, ax1,sfig,unit,key, data_path, True)
            result['c_name'] = title[result.iloc[0]['name']][0]
            result['name'] = title[result.iloc[0]['name']][1]
            if results.empty:
                results = result
            else:
                results = pd.concat([results, result], ignore_index=True)
        else:
            init_background(map1, ax1,sfig, unit,key,data_path)
            
        map1.drawparallels(np.linspace(20, 50, 7),
                           labels=[1, 0, 0, 0],
                           linewidth=0,
                           fontsize=21)
        map1.drawmeridians(np.linspace(75, 135, 7),
                           labels=[0, 0, 0, 1],
                           linewidth=0,
                           fontsize=21)
        
        if len(sfigs.shape)==1:
            ax2 = sfig.add_axes([0.764, 0.088, 0.15, 0.3])
        else:
            ax2 = sfig.add_axes([0.764, 0.088, 0.15, 0.3])
            
        map2 = Basemap(llcrnrlon=106,
                       llcrnrlat=2,
                       urcrnrlon=124,
                       urcrnrlat=23,
                       ax=ax2)
        if language == 'en':
            ax2.text(
                0.98,
                0.02,
                horizontalalignment='right',
                verticalalignment='bottom',
                s=f"South China Sea",
                transform=ax2.transAxes,
                fontsize='xx-large',
            )
            
        init_background(map2, ax2, sfig, unit, key)
        del map1, map2
    plt.show()
    
    # fig.savefig(output,dpi=fig.dpi,bbox_inches='tight')
    # if not results.empty:
    #     results.to_csv(r"", index_label=False)
if __name__=="__main__":
    
    # this is the keys for every mapping, we can select specific keys every time
    title_keys = np.array([
                           "05", "06", "07", "08", "09",
                           "slope_05", "slope_06", "slope_07", "slope_08", "slope_09", "slope_whole",
                           "HWF", "HWMHI", "HWMD", "slope_HWF", "slope_HWMHI", "slope_HWMD",
                           "avg_hazard","slope_hazard", "avg_hazard_Yin",
                           "2013_HWF", "2013_HWMHI", "2013_HWMD", "2018_HWF", "2018_HWMHI", "2018_HWMD", 
                          ])

    #this is the legend display interval for each variables
    title = {
              "05": 10, "06": 10, "07": 10, "08": 10, "09": 10,
              "slope_05": 0.1, "slope_06": 0.1, "slope_07": 0.1, "slope_08": 0.1, "slope_09": 0.1, "slope_whole": 0.01,
              "HWF": 1, "HWMHI": 4, "HWMD": 3, "slope_HWF": 0.05, "slope_HWMHI": 0.2, "slope_HWMD": 0.1,
              "avg_hazard": 1, "slope_hazard": 1, "avg_hazard_Yin": 0.2,
              "2013_HWF": 2, "2013_HWMHI": 5, "2013_HWMD": 5, "2018_HWF": 2, "2018_HWMHI": 5, "2018_HWMD": 5
            }

    img_path = r"" # the folder path where variables raster data existed
    output_path = r"" # output image path
    
    mapping(img_path, output_path, title_keys=title_keys, title=title, statistics=False, language='en')

