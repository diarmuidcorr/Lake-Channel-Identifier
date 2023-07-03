#This code calculates NDWI values for Sentinel 2A data tiles with starting
#point in a folder adjacent to PRODUCT folder, and prints a .tiff image and
#a Lake Area Distribution Histogram to an output folder adjacent to the same
#folers above. 
#It is based on NDWI code from Mahsa Moussavi https://github.com/mmoussavi/Lake_Detection_Satellite_Imagery.


#import required libraries
import rasterio, glob
import rasterio.features as features
from matplotlib import pyplot as plt
import numpy as np
import scipy.ndimage
from scipy.ndimage import label
from scipy.stats import norm
import skimage
from skimage import morphology
from skimage import measure
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import shape



###############################################################################

#Define Threshold values for NDWI calculations
ndwiL = 0.16

#Sentinel-2 data was downloaded prior to processing, it is possible to carryout this
#process without downloading using Google Earth Engine or Google Cloud Platform.    
#Reading in the bands and Resampling to correct resolution where neccessary
#for Sentinel 2 which uses Bands 3 (Green) and 8 (NIR) for NDWI

for imagePath in glob.glob('../PRODUCT/S2A_MSIL1C_***.SAFE/GRANULE/L1C_**/IMG_DATA/'):
#glob allows correct files to be selected, [2,-2]
#bands 3 & 8 for NDWI
    band3Path = str(glob.glob(imagePath+'*_B03.jp2'))[2:-2]   
    band8Path = str(glob.glob(imagePath+'*_B08.jp2'))[2:-2]

#bands 3 & 11 for NDSI
#bands2 and 3 for rock masks
#bands 10,11 and 2 for cloud masks
#bands 3, 4, 2 and 11 for lake masks
    band2Path = str(glob.glob(imagePath+'*_B02.jp2'))[2:-2]
    band4Path = str(glob.glob(imagePath+'*_B04.jp2'))[2:-2]
    band10Path = str(glob.glob(imagePath+'*_B10.jp2'))[2:-2]
    band11Path = str(glob.glob(imagePath+'*_B11.jp2'))[2:-2]
#^removes [' and '] from glob outputs which annoy rasterio    

#tile defined for naming purposes
    tile = str(band3Path)[129:-15]
    directory = str(band3Path)[11:-83]
    
        #print(band3Path, band8Path)
    #define bands
    band3 = rasterio.open(band3Path, driver='JP2OpenJPEG') #green
    band8 = rasterio.open(band8Path, driver='JP2OpenJPEG') #nir

    band2 = rasterio.open(band2Path, driver='JP2OpenJPEG') #blue
    band4 = rasterio.open(band4Path, driver='JP2OpenJPEG') #red
    band10 = rasterio.open(band10Path, driver='JP2OpenJPEG') #swirCirrus res=60m
    band11 = rasterio.open(band11Path, driver='JP2OpenJPEG') #swir res=20m


#number of raster columns
    band3.width
    band8.width

#number of raster rows
    band3.height
    band8.height

    print(tile, 'started')

#generate nir and green objects as arrays in float32 format
    green = band3.read().astype('float32')
    nir = band8.read().astype('float32')

#generate remaining objects as arrays in float32 format
    blue = band2.read().astype('float32')
    red = band4.read().astype('float32')
    swirCirrus = band10.read().astype('float32')
    swir = band11.read().astype('float32')

    print('bands read')
#resample original 20m and 60m images to match other bands resolution
#order=0 uses 'nearest interpolation' to resample.
#flost32 gives shape (1, 10980, 10980), swir has shape (1,5490, 5490)
    swirCirrus = scipy.ndimage.zoom(swirCirrus, (1,6,6), order=0)
    swir = scipy.ndimage.zoom(swir, (1,2,2), order=0)
    
    print('bands resampled')
    

###############################################################################
    

#Carry out NDWI & NDSI in order to create Lake, Rock & Seawater and Cloud masks

#ndwi calculation, empty cells or nodata cells are reported as 0
    np.seterr(divide='ignore', invalid='ignore')
    ndwi = np.empty(band8.shape, dtype=rasterio.float32)
#if green+nir = 0, NDWI value is set to 0.1.
#if green+nir =/= 0, NDWI calc occurs
    ndwi = np.where(
        (green+nir)==0,
        0,
        (green-nir)/(green+nir))
#NDWI minimum value = 0.3 (is set in lake mask below)
    print('ndwi done')
#ndsi calculation, empty cells or nodata cells are reported as 0
    np.seterr(divide='ignore', invalid='ignore')
    ndsi = np.empty(band8.shape, dtype=rasterio.float32)
#if green+swir = 0, NDSI value is set to 0
#if green+swir =/= 0, NDSI calc occurs
    ndsi = np.where(
        (green+swir)==0,
        0,
        (green-swir)/(green+swir))
    print('ndsi done')

#create a rock and seawater mask
#Values of bands are 10000 times higher here than in Landsat-8 Code, Ratios undisturbed
#NDSI set to <0.9 efficientlty removes seawater & rocks from results
#Green<4000 & Blue between 0 and 4000 ensures snow doesn't appear in rock crop
    rock = np.empty(band8.shape, dtype=rasterio.float32)
    rock = np.where((ndsi < 0.9) & (blue > 0) & (blue < 4000) & (green < 4000),
                    1, 0)
    rock_mask=rock

#create a cloud mask
#SWIR>1100 and Cirrus> 30 denotes cloud, Blue band adds light cloud & shadows to cloud mask
    cloud = np.empty(band8.shape, dtype=rasterio.float32)
    cloud = np.where(((swirCirrus > 30) & (swir > 1100)
                      & (blue > 6000) & (blue < 9700)),
                     1, 0)
    cloud_mask = np.where(rock_mask==1,
                          0, cloud)

#create a lake mask
#Threshold values are placed on NDWI value which are stated above
#Green-Red and Blue-Green remove cloud shadows and shaded snow areas (and also areas
#which appear slush-like when limits are increased slightly
#swir > 10 removes some extra cloud shadows, blue>4000 removes snow areas from lakes
#Limits have been adjusted to what appears to work best in comparison with RGB images
    lakeL = np.empty(band8.shape, dtype=rasterio.float32)


    lakeL = np.where((ndwi > ndwiL) & ((blue - red)/(blue + red)>0.18) & ((green - red)>800)
                     & ((green - red)< 4000) & ((blue - green)>400),
                     1, 0)
    lakeL_mask = np.where((cloud_mask == 1), 0, lakeL)
    lakeL_mask = np.where((rock_mask == 1), 0, lakeL_mask)
    
#CC is connected components, this step also removes small cluster #in
#skimag.morphology connectivity=3 indicates 26 neighbours
    CC_L = morphology.label(lakeL_mask, connectivity = 3)

#Removing small clusters of pixels
    processingL = morphology.remove_small_objects(CC_L.astype(bool), min_size=2,
                                       connectivity=3).astype('float32')

#black out the small clusters
    lakeL_mask_cc = np.where(processingL == 0, 0, 1)

#Measuring number of objects (e.g. lakes and channels)
    labelsL, numobjects_L = measure.label(lakeL_mask_cc, return_num=True)

    if numobjects_L == 0:
        print('No lakes or channels present, blank output printed for this tile:', tile)
        with rasterio.open('../Output/GeoTIFFs/NoLake_'+tile+'.tiff', 'w', driver='GTiff',
                       width=band8.width, height = band8.height, count=1,
                       dtype=rasterio.float32, nodata=0, transform=band8.transform,
                       crs=band8.crs) as dst:
            dst.write(lakeL_mask_cc.astype(rasterio.float32))
        shutil.move('../PRODUCT/' + directory, '../PRODUCT/Done/' + directory)
        continue
        
#export ndwi image
    with rasterio.open('../Output/GeoTIFFs/Lake_'+tile+'.tiff', 'w', driver='GTiff',
                       width=band8.width, height = band8.height, count=1,
                       dtype=rasterio.float32, nodata=0, transform=band8.transform,
                       crs=band8.crs) as dst:
        dst.write(lakeL_mask_cc.astype(rasterio.float32))

#Convert Raster to polygon shapefiles
    with rasterio.Env():
        with rasterio.open('../Output/GeoTIFFs/Lake_'+tile+'.tiff') as src:
            image = src.read()
            results = ({'properties': {'raster_val': v}, 'geometry': s}
                       for i, (s, v) in enumerate(features.shapes(image, mask=None,
                                                                  transform=band8.transform)))
     
    geomsL = list(results)
    gpd_polygonized_raster = GeoDataFrame.from_features(geomsL, crs=band8.crs)
    gpd_polygonized_raster.to_file('../Output/Shapefiles/Lake_'+tile+'.shp',
                                   driver = 'ESRI Shapefile')
