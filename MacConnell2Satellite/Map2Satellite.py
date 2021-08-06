import pandas as pd
from shapely.geometry import Point, Polygon

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rowwidth', None)


# returns true if the point is in the polygon specified by the points in the passed row
def pointInProjectedPolygon(_pt, sat_row):
    p1 = Point(sat_row['P1X'], sat_row['P1Y'])
    p2 = Point(sat_row['P2X'], sat_row['P2Y'])
    p3 = Point(sat_row['P3X'], sat_row['P3Y'])
    p4 = Point(sat_row['P4X'], sat_row['P4Y'])
    return _pt.within(Polygon([p1, p3, p2, p4]))


if __name__ == '__main__':
    with open('D:\\MacConnell\\Mapping.txt', 'a') as f:
        # f.write('MacFile,Tile1,Tile2,Tile3,Tile4,Tile5,Tile6,Tile7,Tile8,Tile9\n')

        mac_df = pd.read_csv('D:\\MacConnell\\Photos_Original\\MacConnellCoords.csv')
        sat_df = pd.read_csv('D:\\MacConnell\\Photos_Satellite\\SatelliteCoords.csv')
        already_done = pd.read_csv('D:\\MacConnell\\Mapping.txt')

        mapping = {fname: [[None, None, None] for _ in range(3)] for fname in mac_df['Filename']}

        # every MacConnell image maps to 9 satellite images:
        # | 0 | 1 | 2 |
        # | 3 | 4 | 5 |
        # | 6 | 7 | 8 |

        for c, macRow in mac_df.iterrows():
            macFile = macRow['Filename']
            if macFile not in list(already_done['MacFile']):
                f.write(f'{macFile}')
                X = macRow['Coord X']
                Y = macRow['Coord Y']
                # translate the MacConnell point to 9 total points each shifted by 1500 units corresponding to the 9
                # tiles they would be in
                pts = [[Point(X - 1500, Y + 1500), Point(X, Y + 1500), Point(X + 1500, Y + 1500)],
                       [Point(X - 1500, Y),        Point(X, Y),        Point(X + 1500, Y)],
                       [Point(X - 1500, Y - 1500), Point(X, Y - 1500), Point(X + 1500, Y - 1500)]]
                # check all satellite tiles to find which ones contain each of these augmented points
                tile = None
                for i, lst in enumerate(pts):
                    for j, pt in enumerate(lst):
                        for _, satRow in sat_df.iterrows():
                            if pointInProjectedPolygon(pt, satRow):
                                if tile:
                                    tile = f'({tile}&{satRow["Filename"]})'
                                else:
                                    tile = satRow['Filename']
                        if tile:
                            f.write(f',{tile}')
                        else:
                            print(f'{macFile} at {pt.x, pt.y} could not find a match for tile ({i},{j})')
                            f.write(f',{None}')
                        tile = None
                f.write('\n')
            print(f'{c / len(mac_df) * 100}% done')
