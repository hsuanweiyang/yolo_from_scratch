import numpy as np
import pandas as pd
import cv2
from sys import argv


def modified_data(file_name):

    raw_input = pd.read_csv(file_name)
    X = raw_input[list(raw_input)[1:]]
    X = X.values.reshape(X.shape[0], 28, 28, 1)
    Y = pd.get_dummies(raw_input['label'])
    return X, Y


def generate_img(src_file, num_img, output_dir):
    img_array, label = modified_data(src_file)
    resize_ratio_list = np.arange(0.5, 2.75, 0.25)
    img_label_loc_size = []
    for num in range(num_img):
        background = np.zeros((224,224), dtype=np.float)
        picture_idx = np.random.choice(img_array.shape[0], size=4, replace=False)
        idx = 0
        for r_offset in [0, 112]:
            for c_offset in [0, 112]:
                sub_img = np.squeeze(img_array[picture_idx[idx]]).astype(np.float)
                resize_ratio = np.random.choice(resize_ratio_list, size=1)[0]
                if resize_ratio > 1:
                    interpolation = cv2.INTER_CUBIC
                else:
                    interpolation = cv2.INTER_AREA
                resize_sub_img = cv2.resize(sub_img, (int(sub_img.shape[0]*resize_ratio), int(sub_img.shape[1]*resize_ratio)),
                                            interpolation=interpolation)
                r_range = range(r_offset, r_offset+112-resize_sub_img.shape[0])
                c_range = range(c_offset, c_offset+112-resize_sub_img.shape[1])
                r = np.random.choice(r_range, size=1)[0]
                c = np.random.choice(c_range, size=1)[0]
                background[r:r+resize_sub_img.shape[0], c:c+resize_sub_img.shape[1]] = resize_sub_img
                img_label_loc_size.append(['M{:04d}\t1'.format(num), c-0.5+resize_sub_img.shape[0]//2, r-0.5+resize_sub_img.shape[1]//2,
                                           resize_sub_img.shape[1], resize_sub_img.shape[0]] + list(label.iloc[picture_idx[idx],:]))
                idx += 1
        cv2.imwrite(output_dir + '/M{:04d}.jpg'.format(num), background)
    with open(output_dir + '/label_info.txt', 'w') as output_file:
        for line in img_label_loc_size:
            output_file.write('\t'.join(map(str, line)))
            output_file.write('\n')

if __name__ == '__main__':
    source_file = argv[1]
    num_img = int(argv[2])
    output_dir = argv[3]
    generate_img(source_file, num_img, output_dir)