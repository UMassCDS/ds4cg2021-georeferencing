import pandas as pd

links = pd.read_excel('Links.xlsx', engine='openpyxl')
with open('links.txt', 'w') as f:
    for link in links['URL']:
        name = link[65:76]
        f.write(f'{link}\t{name}\n')
