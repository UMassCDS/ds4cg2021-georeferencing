import pandas as pd
from matplotlib import pyplot as plt
import cv2


pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def make_better_csv():
    distances_df = pd.read_csv('Data\\PropagationDistances.csv')
    better_df = pd.DataFrame(columns=['Source', 'Target', 'PtDistance', 'GCPDistance'])
    for _, row in distances_df.iterrows():
        for i in range(1, 9):
            if row[f'GCPDistance{i}'] != -1:
                better_df = better_df.append({'Source': row['Center'].split('\\')[-1],
                                              'Target': row[f'Tile{i}'].split('\\')[-1],
                                              'PtDistance': float(row[f'PtDistance{i}']),
                                              'GCPDistance': float(row[f'GCPDistance{i}'])}, ignore_index=True)

    better_df.sort_values(by='GCPDistance', inplace=True, ascending=False)
    better_df.to_csv('Data\\PropagationDistancesImproved.csv', index=False)


def add_annotations_csv():
    distances = pd.read_csv('Data\\PropagationDistancesImproved.csv')
    annotations = pd.read_csv('Data\\MacConnellCoords.csv')

    source = []
    target = []
    for c, dist_row in distances.iterrows():
        for _, ann_row in annotations.iterrows():
            if dist_row['Source'].split('-')[-2] == ann_row['Filename'].split('\\')[-1].split('-')[-3]:
                source.append(ann_row['Annotated'])

            if dist_row['Target'].split('-')[-2] == ann_row['Filename'].split('\\')[-1].split('-')[-3]:
                target.append(ann_row['Annotated'])

        print(f'{c / len(distances) * 100}% done')

    distances['Source Annotated'] = source
    distances['Target Annotated'] = target
    distances.to_csv('Data\\PropagationDistancesImproved.csv', index=False)


def scatter_plot_annotated():
    distances = pd.read_csv('Data\\PropagationDistancesImproved.csv')
    x = []
    y = []
    for _, row in distances.iterrows():
        if (row['Source Annotated'] == 1 or row['Source Annotated'] == 2) and (row['Target Annotated'] == 1 or row['Target Annotated'] == 2):
            x.append(row['PtDistance'])
            y.append(row['GCPDistance'])
    plt.scatter(x, y)
    plt.title('Point distance vs. average GCP distance for images that are both annotated')
    plt.xlabel('Point Distance')
    plt.ylabel('Average GCP Distance')
    plt.show()


def scatter_plot_nonannotated():
    distances = pd.read_csv('Data\\PropagationDistancesImproved.csv')
    x = []
    y = []
    for _, row in distances.iterrows():
        if row['Source Annotated'] == 0 and row['Target Annotated'] == 0:
            x.append(row['PtDistance'])
            y.append(row['GCPDistance'])
    plt.scatter(x, y)
    plt.title('Point distance vs. average GCP distance for images that are both not annotated')
    plt.xlabel('Point Distance')
    plt.ylabel('Average GCP Distance')
    plt.show()


def scatter_plot_opposite():
    distances = pd.read_csv('Data\\PropagationDistancesImproved.csv')
    x = []
    y = []
    for _, row in distances.iterrows():
        if (row['Source Annotated'] == 0 and (row['Target Annotated'] == 1 or row['Target Annotated'] == 2)) or (row['Target Annotated'] == 0 and (row['Source Annotated'] == 1 or row['Source Annotated'] == 2)):
            x.append(row['PtDistance'])
            y.append(row['GCPDistance'])
    plt.scatter(x, y)
    plt.title('Point distance vs. average GCP distance for images where one is annotated and the other is not')
    plt.xlabel('Point Distance')
    plt.ylabel('Average GCP Distance')
    plt.show()


def view_picture_pair(stub1, stub2):
    source, target = None, None
    with open('D:\\MacConnell\\Photos_Georeferenced_Samples\\filenames.txt', 'r') as refs:
        for ref in refs.readlines():
            if stub1 in ref:
                source = f'D:\\MacConnell\\Photos_Georeferenced_Samples\\{ref[:-1]}'
            if stub2 in ref:
                target = f'D:\\MacConnell\\Photos_Georeferenced_Samples\\{ref[:-1]}'

        source = cv2.imread(source)
        target = cv2.imread(target)

        plt.axis('off')
        plt.subplot(1, 2, 1)
        plt.imshow(source)
        plt.subplot(1, 2, 2)
        plt.imshow(target)
        plt.show()


def diff_histogram():
    real = []
    df = pd.read_csv('D:\\MacConnell\\PropagationDistances.csv')
    for val in df['Difference']:
        if val != -1:
            real.append(val)
    plt.hist(real, 100)
    plt.title('% Difference of average GCP distance of swapped source/target MacConnell images')
    plt.xlabel('% Difference')
    plt.ylabel('# of occurrences')
    plt.show()


if __name__ == '__main__':
    # view_picture_pair('cni6h11', 'cni6h10')  # example of nothing going on to match
    # view_picture_pair('cni3h90', 'cni3h92')  # 92 -> 90 = 3581m, 90 -> 92 = 37m
    # view_picture_pair('cni6h9', 'cni6h8')  # same as above
    # view_picture_pair('cni4h124', 'cni1h6')  # overlapping, but barely
    # view_picture_pair('dpb1h25', 'dpb4h110')  # overlapping decently, but nothing to match on
    # view_picture_pair('cni3h8', 'cni3h87')
    # diff_histogram()
    # scatter_plot_annotated()
    # scatter_plot_nonannotated()
    scatter_plot_opposite()
