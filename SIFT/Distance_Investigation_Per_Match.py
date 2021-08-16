import numpy as np

if __name__ == '__main__':
    with open('Data/PropagationDistancesPerMatch.csv', 'r') as f:
        matches = {(item[0], item[1]): {'PtDistance': item[2], 'NumMatches': item[3],
                   'Matches': item[4:]} for item in [line.split(',') for line in f.readlines()]}
        for k, v in matches:
            matches[k]['Mean'] = np.mean(v['Matches'])
            matches[k]['Mode'] = np.mode(v['Matches'])
            matches[k]['Median'] = np.median(v['Matches'])
            matches[k]['Std Dev'] = np.std(v['Matches'])

    print(dict(sorted(matches.items(), key=lambda x: x['Mean'])))
