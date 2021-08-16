import pyproj


# function to translate pixel coordinates from a TIFF image to coords in the images epsg projection
def pixel2coord(img, col, row):
    """Returns global coordinates to pixel center using base-0 raster index"""
    # unravel GDAL affine transform parameters
    c, a, b, f, d, e = img.GetGeoTransform()
    xp = a * col + b * row + a * 0.5 + b * 0.5 + c
    yp = d * col + e * row + d * 0.5 + e * 0.5 + f
    return xp, yp


# Converting the epsg:26986 to gps which is epsg:4326
def maccoord2latlon(x1, y1):
    wgs84 = pyproj.Proj(projparams='epsg:4326')
    InputGrid = pyproj.Proj(projparams='epsg:26986')
    return pyproj.transform(InputGrid, wgs84, x1, y1)


# the satellite vectors are in epsg:6347, converting it to that of mac for consistency
def modcoord2latlon(x1, y1):
    mac_coord = pyproj.Proj(projparams='epsg:4326')
    mod_coord = pyproj.Proj(projparams='epsg:6347')
    return pyproj.transform(mod_coord, mac_coord, x1, y1)
