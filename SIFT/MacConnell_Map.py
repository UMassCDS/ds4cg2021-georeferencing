import pandas as pd
from pyproj import Transformer
from geopy import distance as dst
import sys


pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)


if __name__ == '__main__':
    with open('Data\\MacConnellMapping.csv', 'w') as f:
        with open('D:\\MacConnell\\Photos_Georeferenced_Samples\\filenames.txt', 'r') as refs:
            f.write(
                'Center,Tile1,Distance1,Tile2,Distance2,Tile3,Distance3,Tile4,Distance4,'
                'Tile5,Distance5,Tile6,Distance6,Tile7,Distance7,Tile8,Distance8\n')
            trans = Transformer.from_crs('epsg:26986', 'epsg:4326')
            mac_df = pd.read_csv('D:\\MacConnell\\Photos_Original\\MacConnellCoords.csv')
            mapping = {f'D:\\MacConnell\\Photos_Georeferenced_Samples\\{fname[:-1]}':
                       [(None, sys.maxsize) for _ in range(8)] for fname in refs.readlines()}

            c = 0
            for _, row in mac_df.iterrows():
                center = row['Filename']
                stub = center.split('\\')[-1].split('-')[-3]
                for ref in mapping.keys():
                    if stub == ref.split('\\')[-1].split('-')[-2]:
                        center = ref
                if center in mapping.keys():
                    c += 1
                    f.write(f'{center}')
                    cen_pt = trans.transform(row['Coord X'], row['Coord Y'])

                    for _, other_row in mac_df.iterrows():
                        tile = other_row['Filename']
                        stub = tile.split('\\')[-1].split('-')[-3]
                        for ref in mapping.keys():
                            if stub == ref.split('\\')[-1].split('-')[-2]:
                                tile = ref
                        if tile != center and tile in mapping.keys():
                            pt = trans.transform(other_row['Coord X'], other_row['Coord Y'])
                            dist = dst.distance(cen_pt, pt).m
                            for i, near in enumerate(mapping[center]):
                                if near[1] > dist and dist < 4000:
                                    mapping[center][i] = (tile, dist)
                                    mapping[center].sort(reverse=True, key=lambda x: x[1])
                                    break
                    for tile in sorted(mapping[center], key=lambda x: x[1]):
                        f.write(f',{tile[0]},{tile[1]}')
                    f.write('\n')

                    print(f'{c / len(mapping) * 100}% done')
