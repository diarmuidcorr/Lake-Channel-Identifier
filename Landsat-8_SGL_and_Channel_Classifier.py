#This code calculates NDWI values for Landsat 8 data tiles with
#starting point in a folder adjacent to PRODUCT folder, and prints a .tiff image
#and a Lake Area Distribution Histogram to an output folder adjacent to the same
#folers above. 
#It is based on NDWI code from Mahsa Moussavi https://github.com/mmoussavi/Lake_Detection_Satellite_Imagery.
 

#import required libraries
import rasterio, glob
import rasterio.features as features
from matplotlib import pyplot as plt
import numpy as np
import scipy.ndimage
from scipy.ndimage.morphology import binary_closing
from scipy.stats import norm
import skimage
from skimage import morphology, measure
import math
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import shape


###############################################################################

#Define Threshold values for NDWI calculations
ndwiL = 0.16

#Landsat-8 data was downloaded prior to processing, it is possible to carryout this
#process without downloading using Google Earth Engine or Google Cloud Platform.
#Reading in the bandsand Resampling to correct resolution where neccessary
#for Landsat 8 which uses Bands 3 (Green) and 5 (NIR) for NDWI

for imagePath in glob.glob('../LC08_***/LC**/'):
    band2Path = str(glob.glob(imagePath+'*_B2.TIF'))[2:-2]
    band3Path = str(glob.glob(imagePath+'*_B3.TIF'))[2:-2]
    band4Path = str(glob.glob(imagePath+'*_B4.TIF'))[2:-2]#glob allows correct files to be selected, [2,-2]
    band5Path = str(glob.glob(imagePath+'*_B5.TIF'))[2:-2] #^removes [' and '] from glob outputs which annoy rasterio
    band6Path = str(glob.glob(imagePath+'*_B6.TIF'))[2:-2]
    band8Path = str(glob.glob(imagePath+'*_B8.TIF'))[2:-2]
    band10Path = str(glob.glob(imagePath+'*_B10.TIF'))[2:-2]
    
    tile = str(band2Path)[66:-7]
    directory = str(band3Path)[3:-70]
    print(tile, 'started')

    metadataPath = open(str(glob.glob(imagePath+'*_MTL.txt'))[2:-2], 'r')
    metadata = metadataPath.readlines()
    metadataPath.close()    
    

###############################################################################
    


#read in bands, carry out DN to TOA reflectance conversion (needs corrected for solar angle
#& pan-sharpen to 15m (pan sharpen later)
#ρλ = TOA planetary reflectance, θSE = Local sun elevation angle,
#θSZ = Local solar zenith angle > θSZ = 90° – θSE.

#very clumpy, seek improved version
    for line in metadata:
            if 'REFLECTANCE_MULT_BAND_2' in line:
                MULT_BAND_2 = line.split('=')[-1]
                MULT_BAND_2 = MULT_BAND_2.strip('=')
                MULT_BAND_2 = float(MULT_BAND_2)
            if 'REFLECTANCE_ADD_BAND_2' in line:
                ADD_BAND_2 = line.split('=')[-1]
                ADD_BAND_2 = ADD_BAND_2.strip('=')
                ADD_BAND_2 = float(ADD_BAND_2)

            if 'REFLECTANCE_MULT_BAND_3' in line:
                MULT_BAND_3 = line.split('=')[-1]
                MULT_BAND_3 = MULT_BAND_3.strip('=')
                MULT_BAND_3 = float(MULT_BAND_3)
            if 'REFLECTANCE_ADD_BAND_3' in line:
                ADD_BAND_3 = line.split('=')[-1]
                ADD_BAND_3 = ADD_BAND_3.strip('=')
                ADD_BAND_3 = float(ADD_BAND_3)

            if 'REFLECTANCE_MULT_BAND_4' in line:
                MULT_BAND_4 = line.split('=')[-1]
                MULT_BAND_4 = MULT_BAND_4.strip('=')
                MULT_BAND_4 = float(MULT_BAND_4)
            if 'REFLECTANCE_ADD_BAND_4' in line:
                ADD_BAND_4 = line.split('=')[-1]
                ADD_BAND_4 = ADD_BAND_4.strip('=')
                ADD_BAND_4 = float(ADD_BAND_4)

            if 'REFLECTANCE_MULT_BAND_5' in line:
                MULT_BAND_5 = line.split('=')[-1]
                MULT_BAND_5 = MULT_BAND_5.strip('=')
                MULT_BAND_5 = float(MULT_BAND_5)
            if 'REFLECTANCE_ADD_BAND_5' in line:
                ADD_BAND_5 = line.split('=')[-1]
                ADD_BAND_5 = ADD_BAND_5.strip('=')
                ADD_BAND_5 = float(ADD_BAND_5)

            if 'REFLECTANCE_MULT_BAND_6' in line:
                MULT_BAND_6 = line.split('=')[-1]
                MULT_BAND_6 = MULT_BAND_6.strip('=')
                MULT_BAND_6 = float(MULT_BAND_6)
            if 'REFLECTANCE_ADD_BAND_6' in line:
                ADD_BAND_6 = line.split('=')[-1]
                ADD_BAND_6 = ADD_BAND_6.strip('=')
                ADD_BAND_6 = float(ADD_BAND_6)
                
            if 'REFLECTANCE_MULT_BAND_8' in line:
                MULT_BAND_8 = line.split('=')[-1]
                MULT_BAND_8 = MULT_BAND_8.strip('=')
                MULT_BAND_8 = float(MULT_BAND_8)   
            if 'REFLECTANCE_ADD_BAND_8' in line:
                ADD_BAND_8 = line.split('=')[-1]
                ADD_BAND_8 = ADD_BAND_8.strip('=')
                ADD_BAND_8 = float(ADD_BAND_8)

#band 10 requires Radiance bands to be imported and thermal contants are applied
            if 'RADIANCE_MULT_BAND_10' in line:
                MULT_BAND_10 = line.split('=')[-1]
                MULT_BAND_10 = MULT_BAND_10.strip('=')
                MULT_BAND_10 = float(MULT_BAND_10)
            if 'RADIANCE_ADD_BAND_10' in line:
                ADD_BAND_10 = line.split('=')[-1]
                ADD_BAND_10 = ADD_BAND_10.strip('=')
                ADD_BAND_10 = float(ADD_BAND_10)

            if 'K2_CONSTANT_BAND_10' in line:
                K2 = line.split('=')[-1]
                K2 = K2.strip('=')
                K2 = float(K2)
            if 'K1_CONSTANT_BAND_10' in line:
                K1 = line.split('=')[-1]
                K1 = K1.strip('=')
                K1 = float(K1)

            if 'SUN_ELEVATION' in line:
                SUN_ELEVATION = line.split('=')[-1]
                SUN_ELEVATION = SUN_ELEVATION.strip('=')
                SUN_ELEVATION = float(SUN_ELEVATION)
                
    band2 = rasterio.open(band2Path) #blue
    band3 = rasterio.open(band3Path) #green
    band4 = rasterio.open(band4Path) #red
    band5 = rasterio.open(band5Path) #nir
    band6 = rasterio.open(band6Path) #swir
    band8 = rasterio.open(band8Path) #panchromatic
    band10 = rasterio.open(band10Path) #lwir
    
#generate nir and green objects as arrays in float32 format
    blue = band2.read().astype('float32')
    green = band3.read().astype('float32')
    red = band4.read().astype('float32')
    nir = band5.read().astype('float32')
    swir = band6.read().astype('float32')
    panc = band8.read().astype('float32')
    lwir1 = band10.read().astype('float32')
    print('bands read')

#resample bands so that they are in 15m Resolution (like panc)
#by by bilinear interpolation (i.e. order=1), 18241/9121 as blue, green,
#nir, swir have shape 1, 9121, 9121 and panc has shape 1, 18241, 18241
    np.seterr(divide='ignore', invalid='ignore')
    blue = scipy.ndimage.zoom(blue, (1, (18241/9121),(18241/9121)), order=1)
    green = scipy.ndimage.zoom(green, (1,(18241/9121),(18241/9121)), order=1)
    red = scipy.ndimage.zoom(red, (1,(18241/9121),(18241/9121)), order=1)
    nir = scipy.ndimage.zoom(nir, (1,(18241/9121),(18241/9121)), order=1)
    swir = scipy.ndimage.zoom(swir, (1,(18241/9121),(18241/9121)), order=1)
    lwir1 = scipy.ndimage.zoom(lwir1, (1,(18241/9121),(18241/9121)), order=0)
    print('resampled')



###############################################################################
    

#Pansharpen method to improve spectral resolution of multispectral bands using
#the high resolution (15 m) Panchromatic band

#Pansharpening is carried out by Intensity Hue Saturation (IHS) method
    sub = (1/4) * (blue + green + red + nir)
    red = red + (panc - sub)
    green = green + (panc - sub)
    blue = blue + (panc - sub)
    nir = nir + (panc - sub)
    print('Pansharpened')
    

#convert DN to TOA reflectance for each band    
    blue = ((MULT_BAND_2*blue + ADD_BAND_2)/(math.sin(SUN_ELEVATION * math.pi / 180)))
    green = ((MULT_BAND_3*green + ADD_BAND_3)/(math.sin(SUN_ELEVATION * math.pi / 180)))
    red = ((MULT_BAND_4*red + ADD_BAND_4)/(math.sin(SUN_ELEVATION * math.pi / 180)))
    nir = ((MULT_BAND_5*nir + ADD_BAND_5)/(math.sin(SUN_ELEVATION * math.pi / 180)))
    swir = ((MULT_BAND_6*swir + ADD_BAND_6)/(math.sin(SUN_ELEVATION * math.pi / 180)))
    panc = ((MULT_BAND_8*panc + ADD_BAND_8)/(math.sin(SUN_ELEVATION * math.pi / 180)))
    
#for band 10 Radiance, rather than reflectance is used,
#it then has thermal constants K1 and K2 applied
    lwir1 = lwir1*MULT_BAND_10 + ADD_BAND_10
    lwir1 = K2/(np.log((K1/lwir1)+1))
    print('toa done')

    
###############################################################################
    

#Carry out NDWI, TIRS/Blue & NDSI in order to create Lake, Rock & Seawater
#and Cloud masks
     
#ndwi calculation, empty cells or nodata cells are reported as 0
    np.seterr(divide='ignore', invalid='ignore')
    ndwi = np.empty(band8.shape, dtype=rasterio.float32)
    ndwi=np.where(
        (green+nir)==0.,
        0,
        (green-nir)/(green+nir))
#NDWI minimum value = 0.3 & NDWI maximum value = 0.8
    print('ndwi done')

#ndsi calculation, empty cells or nodata cells are reported as 0
    ndsi = np.empty(band8.shape, dtype=rasterio.float32)

    ndsi = np.where(
        (green+swir)==0,
        0,
        (green-swir)/(green+swir))

    print('ndsi done')

#Calculation of TIRS/Blue
    TIRS_Blue = np.empty(band8.shape, dtype=rasterio.float32)

    TIRS_Blue = np.where(
        blue==0,
        0, (lwir1/blue))
    print('TIRS-Blue done')

#Rock outcrop/Seawater Masking (TIRS_Blue > 0.65 removes exposed rock,
#red>0 removes border interference, blue < 0.35 removes snow from rock mask)
    rock = np.empty(band8.shape, dtype=rasterio.float32)
    rock = np.where((TIRS_Blue > 650), 1, 0)
    rock = np.where((red > 0), 1, 0)
    rock = np.where((blue < 0.35),1,0)
    rock_mask = rock
    
#create a cloud mask (SWIR>0.1 denotes cloud, ndsi <0.8 removes most cloud,
#Blue band restricts light cloud and cloud shadows)
    cloud = np.empty(band8.shape, dtype=rasterio.float32)
    cloud = np.where((swir > 0.1) & (ndsi < 0.8) & (blue > 0.6) & (blue < 0.95),
                     1, 0)
    cloud_mask = cloud

#create a lake mask (Threshold values are placed on NDWI value which are stated above
#Green-Red and Blue-Green remove cloud shadows and shaded snow areas (and also areas
#which appear slush-like when limits are increased slightly)
#Green-red and blue-green slightly lower here than in S2 as this ruled out
#blue ice and shaded areas which were more evident here than in S2
    lakeL = np.empty(band8.shape, dtype=rasterio.float32)
    lakeL = np.where((ndwi > ndwiL) & ((green - red) > 0.08) & ((green - red)< 0.4) &
                     ((blue - green) > 0.040)  & ((blue - red)/(blue + red)>0.18),
                     1, 0)

    lakeL_mask = np.where((cloud_mask == 1),
                         0, lakeL)
    lakeL_mask = np.where((rock_mask == 1),
                         0, lakeL)


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
        shutil.move('../' + directory, '../Done/' + directory)
        continue
        
        
#export ndwi image       
    with rasterio.open('../Output/GeoTIFFs/Lakes_'+tile+'.tiff', 'w', driver='GTiff',
                       width=band8.width, height = band8.height, count=1,
                       dtype=rasterio.float32, nodata=0, transform=band8.transform,
                       crs=band8.crs) as dst:
         dst.write(lakeL_mask_cc.astype(rasterio.float32))

    print('Image saved')

#Convert Raster to polygon shapefiles
    with rasterio.Env():
        with rasterio.open('../Output/GeoTIFFs/Lakes_'+tile+'.tiff') as src:
            image = src.read()
            results = ({'properties': {'raster_val': v}, 'geometry': s}
                       for i, (s, v) in enumerate(features.shapes(image, mask=None,
                                                                  transform=band8.transform)))
            src.closed
    geomsL = list(results)
    gpd_polygonized_raster = GeoDataFrame.from_features(geomsL, crs=band8.crs)
    gpd_polygonized_raster.to_file('../Output/Shapefiles/Lakes_'+tile+'.shp',
                                   driver = 'ESRI Shapefile')
