import pandas as pd
import numpy as np
from sys import argv
import os
import xml.etree.cElementTree as et

input_dir = argv[1]
outfile_name = argv[2]
all_data = {}

for file in os.listdir(input_dir):
    tree = et.ElementTree(file=os.path.join(input_dir, file))
    root = tree.getroot()
    all_data[file[:-4]] = []
    for ob in root.findall('object'):
        one_hot = list(map(int, np.zeros(10)))
        one_hot[int(ob.find('name').text)] = 1
        class_label = one_hot
        coordinate = {'label': class_label}
        for i in ob.find('bndbox'):
            coordinate[i.tag] = int(i.text)
        coordinate['w'] = coordinate['xmax'] - coordinate['xmin']
        coordinate['h'] = coordinate['ymax'] - coordinate['ymin']
        coordinate['x'] = coordinate['xmin'] + coordinate['w']/2
        coordinate['y'] = coordinate['ymin'] + coordinate['w']/2

        for i in ob.find('bndbox'):
            del coordinate[i.tag]
        all_data[file[:-4]].append(coordinate)

with open(outfile_name, 'w') as output_file:
    for file_name in all_data:
        for data in all_data[file_name]:
            out = map(str,[file_name, 1, data['x'], data['y'], data['w'], data['h'], *data['label']])
            line = '\t'.join(list(out)) + '\n'
            output_file.write(line)

