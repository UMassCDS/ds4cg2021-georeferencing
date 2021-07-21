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

    return R * c


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
