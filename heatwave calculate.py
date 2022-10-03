# calculate TI, HI, and Heatwave indicators

import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from osgeo import gdal
import numpy.ma as ma
from scipy import stats

def openSingleImage(filename: str):
    """ Open one image

    Args:
        filename (str): 文件路径

    Returns:
        im_data: array,数据数组
        im_proj: str,坐标系
        im_geotrans: tuple,仿射矩阵
        im_height,im_width: float,图像高和宽
        ndv: float,NoDataValue
    """

    dataset = gdal.Open(filename)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_band = dataset.GetRasterBand(1)
    ndv = im_band.GetNoDataValue()
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    del dataset
    return im_data, im_proj, im_geotrans, im_height, im_width, ndv
  
def openImages(filenames):
    """
    打开工作文件夹中的所有影像
    dirpath:数据读取文件夹
    return:Igs,im_geotrans.im_proj,ndv
    """
    Igs = []
    idate = 0
    with tqdm(filenames) as t:
        for filename in t:
            if filename[-4:] == ".tif":
                Image, im_proj, im_geotrans, _, _, ndv = openSingleImage(
                    filename)
                Igs.append(Image)
                idate = idate + 1
                t.set_description(filename + " is already open……")
    return np.array(Igs), im_geotrans, im_proj, ndv
  
def write_img(im_data, filename, im_proj, im_geotrans, dirpath, ndv):
    """写影像"""
    # im_data 被写的影像
    # im_proj, im_geotrans 均为被写影像参数
    # filename 创建新影像的名字，dirpath 影像写入文件夹
    
    # 判断栅格数据类型
    datatype = gdal.GDT_Float32
    fullpath = dirpath + "\\" + filename
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_bands = 1  # 均为单波段影像
        im_height, im_width = im_data.shape
        im_data = [im_data]
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(fullpath, im_width, im_height, im_bands, datatype)
    if dataset != None:
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).SetNoDataValue(ndv)  # 设置nodata值
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    else:
        os.remove(fullpath)
        with open(r"I:/dem_chazhi/log/error.txt", "a") as o:
            o.write(fullpath + "\r")
    del dataset
    del im_data

# calculate TI
def calc_torridity_index(RHU, TEM):

    minRHU = ma.masked_where(RHU > 60, RHU).filled(0)
    minTEM = ma.masked_where(RHU > 60, TEM).filled(0)
    maxRHU = ma.masked_where(RHU <= 60, RHU).filled(0)
    maxTEM = ma.masked_where(RHU <= 60, TEM).filled(0)
    minTI = 1.8 * minTEM - 0.55 * (1.8 * minTEM - 26) * (1 - 60 * 0.01) + 32
    maxTI = 1.8 * maxTEM - 0.55 * (1.8 * maxTEM - 26) * (1 -
                                                         maxRHU * 0.01) + 32
    TI = maxTI + minTI
    return TI

# calculate TI series  
def calc_TIs():
    global tems_path
    global rhus_path
    global tem_files
    turn_path = r"I:\dem_chazhi\result\TI"
    with tqdm(tem_files) as t:
        for tem_file in t:
            t.set_description_str("Done:")
            TI_file = f"TI_{tem_file[-12:]}"
            if not os.path.exists(os.path.join(turn_path, TI_file)):
                rhu_file = f"{rhus_path}/RHU-13003_{tem_file.split('/')[-1][10:]}"
                tem_img, im_proj, im_geotrans, im_height, im_width, ndv = openSingleImage(
                    tem_file)
                rhu_img = openSingleImage(rhu_file)[0]
                mask = (tem_img == ndv)
                tem_mask = ma.masked_where(mask, tem_img)
                rhu_mask = ma.masked_where(mask, rhu_img)
                TI = calc_torridity_index(rhu_mask, tem_mask)
                TI = ma.masked_where(mask, TI).filled(ndv)
                write_img(TI, TI_file, im_proj, im_geotrans, turn_path, ndv)
                
# calculate the TI series to calculate TI' when tem_img <= T_threshold based on definition 
def calc_torridity_threshold(T_threshold=33):
    global tems_path
    global rhus_path
    global tem_files
    turn_path = r"" #
    with tqdm(tem_files) as t:
        for tem_file in t:
            t.set_description_str("Done:")
            TI_threshold_file = f"TI_threshold_{tem_file[-12:]}"
            if not os.path.exists(os.path.join(turn_path, TI_threshold_file)):
                rhu_file = f"{rhus_path}/RHU-13003_{tem_file.split('/')[-1][10:]}"
                tem_img, im_proj, im_geotrans, im_height, im_width, ndv = openSingleImage(
                    tem_file)
                rhu_img = openSingleImage(rhu_file)[0]
                mask = (tem_img == ndv) + (tem_img <= T_threshold)
                tem_mask = ma.masked_where(mask, tem_img)
                rhu_mask = ma.masked_where(mask, rhu_img)
                TI_threshold = calc_torridity_index(rhu_mask, tem_mask)
                TI_threshold = ma.masked_where(mask, TI_threshold).filled(ndv)
                write_img(TI_threshold, TI_threshold_file, im_proj,
                          im_geotrans, turn_path, ndv)

# calculate TI' (nodata value in China will make error in calculating HI)
def calc_TIpie():
    path = r"" # path to store daily TI_threshold 
    split_path = r"" # path to store split raster path because of insufficient computer running memory
    files = [i for i in os.listdir(path) if i.endswith("tif")]

    with tqdm(range(8)) as t1:
        for i in t1:
            t1.set_description_str("Done")
            first = False
            Igs = []
            with tqdm(files) as t2:
                for file in t2:
                    t2.set_description_str("Open")
                    split_file = os.path.join(split_path,
                                              file[:-4] + str(i) + ".tif")
                    if os.path.exists(split_file):
                        im_data, im_proj, im_geotrans, im_height, im_width, ndv = openSingleImage(
                            split_file)
                        im_data = ma.masked_where(im_data == ndv,
                                                  im_data).filled(np.nan)
                        Igs.append(im_data)

            Igs = np.array(Igs)
            out_img = np.nanmedian(Igs, axis=0)
            write_img(out_img, "TI'_" + str(i) + ".tif", im_proj, im_geotrans,
                      r"", ndv)
            
# fill the nodata value with max value (nodata value means no MT>=33 in 30 years)
def calc_TIpie_2():
    TIpie_2 = r"" # path to store filled TIpie
    TIs_path = r"" # folder of TIs
    max_path = r"" # path of max_TI
    TIs_name = [i for i in os.listdir(TIs_path) if i.endswith(".tif")]
   
    TI_mask = os.path.join(TIs_path, TIs_name[0]) # to judge if in China
    im_data, ndv = openSingleImage(TI_mask)[0], openSingleImage(TI_mask)[-1]
    mask = im_data != ndv
    del im_data
    
    # calculate max TI of 30 years
    first = True
    Max = None
    with tqdm(range(8)) as t1:
        for i in t1:
            t1.set_description_str("Done")
            # is_first = True
            Igs = []
            with tqdm(TIs_name) as t2:
                for file in t2:
                    t2.set_description_str("Open")
                    split_file = os.path.join(split_path,
                                              file[:-4] + str(i) + ".tif")
                    if os.path.exists(split_file):
                        data, im_proj, im_geotrans, _, _, ndv = openSingleImage(
                            split_file)
                        Igs.append(data)
            print("Calculating Max:")
            Igs = np.array(Igs)
            max = np.nanmax(Igs, axis=0)
            write_img(max, "Max_" + str(i) + ".tif", im_proj, im_geotrans,
                      "I:/dem_chazhi/result", ndv)
            del Igs
            del max
            if first:
                Max = max
                first = False
            else:
                 Max = np.append(Max, max, axis=1)

    # fill nodata value with Max
    dataset = gdal.Open(max_path)
    im_band = dataset.GetRasterBand(1)
    Max = im_band.ReadAsArray()
    del dataset

    dataset = gdal.Open(TIpie_copy, GA_Update)
    im_band = dataset.GetRasterBand(1)
    im_data = im_band.ReadAsArray()

    (height, width) = im_data.shape
    for i in tqdm(range(height)):
        for j in tqdm(range(width)):
            if mask[i][j] and im_data[i][j] == ndv and Max[i][j] != ndv:
                im_data[i][j] = Max[i][j]
    im_band.WriteArray(im_data)
    del dataset
    
# calculate HI by TI and TI'
def calc_HI():
    years = np.arange(1990, 2000) # this is a changeable variable based on need of oneself
    TI_path = r"" # TI folder
    TIpie_path = r"" # filled TI'
    output_path = r"" # HI folder
    TIpie, im_proj, im_geotrans, im_height, im_width, ndv = openSingleImage(
        TIpie_path)
    mask = TIpie == ndv
    TIpie = ma.masked_where(mask, TIpie)
    for year in years:
        if not os.path.exists(os.path.join(output_path,
                                           f"{str(year)}_HIs.tif")):
            TIs = [
                os.path.join(TI_path, i) for i in os.listdir(TI_path)
                if str(year) in i
            ]
            TIs = openImages(TIs)[0]
            hotimg = np.zeros_like(TIs, dtype=np.bool8)

            hotimg[TIs > TIpie] = True
            hotimg = np.transpose(hotimg)
			
			# HWMD
			longest_duration = np.zeros_like(TIs[0])
			
			# create HIs
			HIs = np.zeros_like(TIs)
			with tqdm(range(im_width)) as t:
                for i in t:
                    for j in range(im_height):
                        if np.max(hotimg[i][j]) == 0:
                            continue
						
						date = 0 # heatwave duration days so far
						for k, is_hot in enumerate(hotimg[i][j]):
                            if is_hot:
                                sum = 0
                                HI = 1.2 * (TIs[k][j][i] - TIpie[j][i])
                                if date >= 1:
                                    for l in range(date):
                                        ndi = l + 1
                                        sum += 0.35 * (
                                            1 / ndi *
                                            (TIs[k - ndi][j][i] - TIpie[j][i])
                                        ) + 0.15 * (1 / ndi)  #1/2=0.5
                                HI += sum + 1
                                HIs[k][j][i] = HI
                                date += 1
                            else:
                                date = 0
			
			for HI in HIs:
				HI = ma.masked_where(mask, HI).filled(ndv)			
			
			write_img(HIs, f"{str(year)}_HIs.tif", im_proj, im_geotrans,
                      output_path, ndv)
            del HIs
            del TIs
            del hotimg
			
