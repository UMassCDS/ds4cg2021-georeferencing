import os
import random
from PIL import Image

loc = "D:\\MacConnell\\Photos_Original\\"
files = []

for i in range(1, 8):
    ext = f'b00{i}\\Annotated\\'
    for infile in os.listdir(loc + ext):
        if infile[-3:] == "tif":
            files.append(loc + ext + infile)

paths = [files[random.randint(0, len(files))] for _ in range(10)]

for path in paths:
    im = Image.open(path)
    im.show()
