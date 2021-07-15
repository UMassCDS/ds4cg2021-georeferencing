import shutil
from pathlib import Path

from PIL import Image
import os
import numpy as np
from skimage import color


# returns the longest distance between consecutive points in the inp strips
# and a count of the distances greater than or equal to thresh
def get_max_diff(inp: [], thresh):
    max_dist, count = 0, 0
    for line in inp:
        for i in range(0, len(line)):
            dist = np.linalg.norm(line[i - 1] - line[i])
            if dist > max_dist:
                max_dist = dist
            if dist >= thresh:
                count += 1
    return max_dist, count


for i in range(1, 8):
    # file locations
    ext = f'b00{i}'
    loc_orig = f'D:\\MacConnell\\Photos_Original\\{ext}\\'
    loc_annotated = f'D:\\MacConnell\\Photos_Original\\{ext}\\Annotated\\'
    loc_nonannotated = f'D:\\MacConnell\\Photos_Original\\{ext}\\NonAnnotated\\'
    for infile in os.listdir(loc_orig):
        if infile[-3:] == "tif":
            print(f'file : {loc_orig}{infile}')
            try:
                orig = Image.open(loc_orig + infile)
                temp = np.asarray(orig)
                # crop out the edges to get rid of off-center borders, black notches, and ID text
                im = orig.crop((250, 500, len(temp) - 250, len(temp[0]) - 500))
                im = np.asarray(im)
                # convert to 'lab'; a different color encoding that is perceptually uniform
                im = color.rgb2lab(im)

                # compare pixel color differences
                lines = []
                for perc in [int(len(im) / 10 * i) for i in range(0, 10)]:
                    lines.append(im[perc])
                for perc in [int(len(im[0]) / 10 * i) for i in range(0, 10)]:
                    lines.append(im[:, perc])

                maxDiff, cnt = get_max_diff(lines, 25)
                print(maxDiff, cnt)
                # convert back to RGB to save the image
                im = color.lab2rgb(im)
                im = Image.fromarray((im * 255).astype(np.uint8))

                if (cnt > 5 and maxDiff < 50) or (cnt > 300 and maxDiff >= 50):
                    # move the file to the annotated folder if it has enough color swings
                    shutil.move(loc_orig + infile, loc_annotated + infile)
                    # move the .xml file if it exists
                    print(loc_orig + infile[:-8] + '.xml')
                    if Path(loc_orig + infile[:-8] + '.xml').is_file():
                        shutil.move(loc_orig + infile[:-8] + '.xml', loc_annotated + infile[:-8] + '.xml')
                    print(f"Annotated and moved to {loc_annotated}{infile}")
                    # add the filename to the list
                    with open('D:\\MacConnell\\Photos_Original\\Annotated.txt', 'a') as f:
                        f.write(f'{ext}\\{infile}\n')
                else:
                    # move the file to the nonannotated folder if it has enough color swings
                    shutil.move(loc_orig + infile, loc_nonannotated + infile)
                    # move the .xml file if it exists
                    print(loc_orig + infile[:-8] + '.xml')
                    if Path(loc_orig + infile[:-8] + '.xml').is_file():
                        shutil.move(loc_orig + infile[:-8] + '.xml', loc_nonannotated + infile[:-8] + '.xml')
                    print(f"NonAnnotated and moved to {loc_nonannotated}{infile}")
                    # add the filename to the list
                    with open('D:\\MacConnell\\Photos_Original\\NonAnnotated.txt', 'a') as f:
                        f.write(f'{ext}\\{infile}\n')

            # the file is corrupted
            except Image.UnidentifiedImageError:
                print(f'failed to categorize {ext}\\{infile}')
