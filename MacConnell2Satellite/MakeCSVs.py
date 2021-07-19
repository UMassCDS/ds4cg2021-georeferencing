import os
import pandas as pd
from xml.dom import minidom
from pyproj import Transformer


sat_loc = 'D:\\MacConnell\\Photos_Satellite\\'
sat = pd.DataFrame(columns=['Filename', 'P1X', 'P1Y', 'P2X', 'P2Y', 'P3X', 'P3Y', 'P4X', 'P4Y', 'Original EPSG',
                            'Current EPSG'])
OTHER_EPSG = pd.read_csv('D:\\MacConnell\\Photos_Satellite\\OtherEPSG.csv')

if __name__ == '__main__':
    for folder in os.listdir(sat_loc):
        if os.path.isdir(sat_loc + folder):
            for file in os.listdir(sat_loc + folder):
                if file[-7:] == 'aux.xml':
                    # read xml file
                    xml = minidom.parse(sat_loc + folder + '\\' + file)
                    # get lower corner coordinates
                    lower = xml.getElementsByTagName('gml:lowerCorner')
                    p1x = int(lower[0].firstChild.data.split()[0])
                    p1y = int(lower[0].firstChild.data.split()[1])
                    # get upper corner coordinates
                    upper = xml.getElementsByTagName('gml:upperCorner')
                    p2x = int(upper[0].firstChild.data.split()[0])
                    p2y = int(upper[0].firstChild.data.split()[1])
                    # get the epsg code to translate from
                    from_epsg = xml.getElementsByTagName('gml:Envelope')[0].attributes['srsName'].value[-4:]
                    transformer = Transformer.from_crs(f'epsg:{from_epsg}', 'epsg:26986')
                    # interpolate the other two corners of the tile while its still a perfect square
                    p3x, p3y = transformer.transform(p1x, p2y)
                    p4x, p4y = transformer.transform(p2x, p1y)
                    # now we can project the original points as well
                    p1x, p1y = transformer.transform(p1x, p1y)
                    p2x, p2y = transformer.transform(p2x, p2y)

                    # append the filename and coords to the csv
                    sat = sat.append({'Filename': sat_loc + folder + '\\' + file[:-8], 'Original EPSG': from_epsg,
                                      'Current EPSG': '26986', 'P1X': p1x, 'P1Y': p1y, 'P2X': p2x, 'P2Y': p2y,
                                      'P3X': p3x, 'P3Y': p3y, 'P4X': p4x, 'P4Y': p4y},
                                     ignore_index=True)

    sat.to_csv(sat_loc + 'SatelliteCoords.csv', index=False)

    # 4996 rows total, 4325 with all four of these columns defined - 671 missing
    mac = pd.read_csv('D:\\MacConnell\\Photos_Original\\Photo List.csv')[['Photo Folder', 'Photo Filename',
                                                                          'Coord X', 'Coord Y']] \
        .dropna(axis=0, how='any')
    temp = pd.read_csv('D:\\MacConnell\\Photos_Original\\Photo List.csv')[['Photo Filename', 'Annotated', 'GCPs']]\
        .set_index('Photo Filename')
    mac = mac.join(temp, how='left', on='Photo Filename')
    # 2 for annotated, 1 for partially annotated, and 0 for not annotated. -1 for unknown
    mac = mac.replace({'yes': 2, 'yes ': 2, 'full': 2, 'partial': 1, 'no': 0, 'none': 0}).fillna(-1)
    # add a column that is 0 or 1 whether or not the image has already been georeferenced
    geo = []
    for _, row in mac.iterrows():
        if row['GCPs'] == -1:
            geo.append(0)
        else:
            geo.append(1)
    mac['GeoReferenced'] = geo
    # make a filename column and strip the Photo List file of unnecessary columns
    mac['Filename'] = 'D:\\MacConnell\\Photos_Original\\' + mac['Photo Folder'] + '\\' + mac['Photo Filename']
    mac = mac[['Filename', 'Coord X', 'Coord Y', 'GeoReferenced', 'Annotated']]
    mac.to_csv('D:\\MacConnell\\Photos_Original\\MacConnellCoords.csv', index=False)
