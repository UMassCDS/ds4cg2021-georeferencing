from osgeo import gdal
from math import sin, cos, sqrt, atan2, radians
import pyproj


def scaling(am_width, am_len):
    width_scale = am_width / 9.0
    len_scale = (9.0 - am_len) / 9.0
    new_x = width_scale * 5463  # historic_raster.width
    new_y = len_scale * 5530  # historic_raster.height
    return new_x, new_y


# function to translate pixel coordinates from a TIFF image to lat/lon coordinates
def pixel2coord(img, col, row):
    """Returns global coordinates to pixel center using base-0 raster index"""
    # unravel GDAL affine transform parameters
    c, a, b, f, d, e = img.GetGeoTransform()
    xp = a * col + b * row + a * 0.5 + b * 0.5 + c
    yp = d * col + e * row + d * 0.5 + e * 0.5 + f
    return xp, yp


def ground_dist(latmac, lonmac, latsat, lonsat):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(latmac)
    lon1 = radians(lonmac)
    lat2 = radians(latsat)
    lon2 = radians(lonsat)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c

    print("Result:", distance)


# Converting the epsg:26986 to gps which is epsg:4326
def coord2latlon(x1, y1):
    wgs84 = pyproj.Proj(projparams='epsg:4326')
    InputGrid = pyproj.Proj(projparams='epsg:26986')
    return pyproj.transform(InputGrid, wgs84, x1, y1)


# the satellite vectors are in epsg:6347, converting it to that of mac for consistency
def mod2maccoord(x1, y1):
    mac_coord = pyproj.Proj(projparams='epsg:26986')
    mod_coord = pyproj.Proj(projparams='epsg:6347')
    return pyproj.transform(mod_coord, mac_coord, x1, y1)


'''
#Testing with GCP from file
#1.309706	0.645108	100526.776638	869451.328514
img=gdal.Open("C:/Users/sowmy/Desktop/DS4CG/GeoReferencing/GeoReferencing/mac_sample/mufs190-1952-cni5h100-i001.reference.tif")
mod = gdal.Open('C:/Users/sowmy/Desktop/DS4CG/GeoReferencing/GeoReferencing/sat_sample/18TXM900605/18TXM900605.jp2',gdal.GA_ReadOnly)
#Convert the lengths to pixel for mac
col,row = scaling(1.309706,0.645108)

#pixel to coordinates for mac
col_cord,row_cord=pixel2coord(img,col,row)

#epsg:26986 -> epsg:4326 which is general latlon convention
latmac,lonmac=coord2latlon(col_cord,row_cord)

latmod,lonmod = coord2latlon(100526.776638,869451.328514)

#Calculating distance on Ground
ground_dist(latmac,lonmac,latmod,lonmod)
'''
