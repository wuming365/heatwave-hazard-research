# Mapping of variables

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

