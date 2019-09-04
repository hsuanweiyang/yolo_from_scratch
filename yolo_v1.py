import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from sys import argv
import os
from collections import defaultdict
import math
#os.environ['KMP_DUPLICATE_LIB_OK']='True'


# global variable
cell_size = 7
num_class = 10
boxes_per_cell = 1
img_size = [224,224]


def read_img(data_dir):
    img_files = os.listdir(data_dir)
    img_data = []
    for img_file in img_files:
        img_data.append(cv2.imread(os.path.join(data_dir, img_file), cv2.IMREAD_GRAYSCALE))
    img_data = np.asarray(img_data)
    img_data = img_data.reshape(img_data.shape[0], img_data.shape[1], img_data.shape[2], 1)
    return img_data


def read_info(file_path):
    label_info_data = []
    with open(file_path, 'r') as input_file:
        line_data = input_file.readlines()
        for line in line_data:
            label_info_data.append(line.strip().split('\t'))
    return label_info_data


def draw_bounding_box(img, label_info):
    for info in label_info:
        confidence = float(info[1])
        x_cen = int(info[2])
        y_cen = int(info[3])
        w = int(info[4])
        h = int(info[5])
        category = np.argmax(info[6:])
        x_min = x_cen - w//2
        x_max = x_cen + w//2
        y_min = y_cen - h//2
        y_max = y_cen + h//2
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0,0,255), 2)
        cv2.putText(img, '{0}:{1}'.format(category, confidence), (x_min, y_min-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,0,255), 1)
        cv2.imwrite('output.jpg', img)


def modify_data(data_dir, label_file):
    label_by_id = defaultdict(list)
    with open(label_file, 'r') as label_input:
        lines = label_input.readlines()
        for line in lines:
            line_data = line.strip().split('\t')
            id = line_data[0]
            label_by_id[id].append(list(map(float, line_data[1:]))) # format: id, confidence, x, y, w, h, class*10

    img_by_id = {}
    for img_id in label_by_id:
        img_data = cv2.imread(os.path.join(data_dir, '{0}.jpg'.format(img_id)), cv2.IMREAD_GRAYSCALE)
        img_by_id[img_id] = img_data.reshape(img_data.shape[0], img_data.shape[1], 1)
    Y = []
    X = []
    for id in img_by_id:
        output = np.zeros((7, 7, 15))
        sub_label_info = label_by_id[id]
        for label in sub_label_info:
            x, y = label[1], label[2]
            grid_x = (x-1)//32     # input dim: 224, output: 7    =>  224/7=32
            grid_y = (y-1)//32
            confidence = label[0]
            relative_x = (((x-1)%32)+1)/32
            relative_y = (((y-1)%32)+1)/32
            relative_w = label[3]/224
            relative_h = label[4]/224
            new_label = np.zeros(15)
            new_label[:5] = [confidence, relative_x, relative_y, relative_w, relative_h]
            new_label[5:] = label[5:]
            output[int(grid_y), int(grid_x), :] = new_label  # format: confidence, x, y, w, h, class*10
        Y.append(output)
        X.append(img_by_id[id])

    return np.asarray(X), np.asarray(Y)

'''
def iou(box1, box2):
    # box: (x, y, w, h) original scale
    box1_x_max = box1[0] + box1[2]//2
    box1_x_min = box1[0] - box1[2]//2
    box1_y_max = box1[1] + box1[3]//2
    box1_y_min = box1[1] - box1[3]//2
    box2_x_max = box2[0] + box2[2]//2
    box2_x_min = box2[0] - box2[2]//2
    box2_y_max = box2[1] + box2[3]//2
    box2_y_min = box2[1] - box2[3]//2
    inter_x_max = min(box1_x_max, box2_x_max)
    inter_x_min = max(box1_x_min, box2_x_min)
    inter_y_max = min(box1_y_max, box2_y_max)
    inter_y_min = max(box1_y_min, box2_y_min)
    inter_area = max(inter_x_max-inter_x_min, 0) * max(inter_y_max-inter_y_min, 0)
    box1_area = (box1_x_max-box1_x_min) * (box1_y_max-box1_y_min)
    box2_area = (box2_x_max-box2_x_min) * (box2_y_max-box2_y_min)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area/union_area
    return iou
'''


def create_placeholder(in_h, in_w, in_c, on_h, on_w, on_c):
    X = tf.placeholder(tf.float32, [None, in_h, in_w, in_c])
    Y = tf.placeholder(tf.float32, [None, on_h, on_w, on_c])
    return X, Y


def initialize_parameter():
    filter_shrink = 4
    w1 = tf.get_variable('w1', [7,7,1,32//filter_shrink], initializer=tf.keras.initializers.he_normal())
    w2 = tf.get_variable('w2', [3,3,32//filter_shrink,64//filter_shrink], initializer=tf.keras.initializers.he_normal())
    w3 = tf.get_variable('w3', [1,1,64//filter_shrink,16//filter_shrink], initializer=tf.keras.initializers.he_normal())
    w4 = tf.get_variable('w4', [3,3,16//filter_shrink,128//filter_shrink], initializer=tf.keras.initializers.he_normal())
    parameters = {'w1':w1, 'w2':w2, 'w3':w3, 'w4':w4}
    return parameters


def forward_propagation(X, parameters, is_training, drop_rate):
    if is_training is False:
        drop_rate = 0.

    w1 = parameters['w1']
    w2 = parameters['w2']
    w3 = parameters['w3']
    w4 = parameters['w4']

    Z1 = tf.nn.conv2d(X, w1, strides=[1,2,2,1], padding='SAME')   # 112
    A1 = tf.layers.batch_normalization(Z1, training=is_training)
    A1 = tf.nn.leaky_relu(A1)

    P1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    # 56

    Z2 = tf.nn.conv2d(P1, w2, strides=[1,1,1,1], padding='SAME') # 56
    A2 = tf.layers.batch_normalization(Z2, training=is_training)
    A2 = tf.nn.leaky_relu(A2)

    P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # 28

    Z3 = tf.nn.conv2d(P2, w3, strides=[1,1,1,1], padding='SAME')
    A3 = tf.layers.batch_normalization(Z3, training=is_training)
    A3 = tf.nn.leaky_relu(A3)

    P3 = tf.nn.max_pool(A3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # 14

    Z4 = tf.nn.conv2d(P3, w4, strides=[1,2,2,1], padding='SAME')   # 7
    A4 = tf.layers.batch_normalization(Z4, training=is_training)
    A4 = tf.nn.leaky_relu(A4)

    conv_out = tf.layers.flatten(A4)

    fc_one = tf.keras.layers.Dense(512, activation=None)
    Z5 = fc_one(conv_out)
    A5 = tf.nn.leaky_relu(Z5)
    A5 = tf.nn.dropout(A5, rate=drop_rate)

    fc_out = tf.keras.layers.Dense(735, activation=None)
    Z_out = fc_out(A5)
    #Z_out = tf.reshape(Z_out, [tf.shape(X)[0],7,7,20])
    return Z_out


def loc_scale_relative_to_origin(relative_loc, grid_x, grid_y, grid_step, img_size):
    ori_x = int((relative_loc[0] + grid_x) * grid_step - 1)
    ori_y = int((relative_loc[1] + grid_y) * grid_step - 1)
    ori_w = int(relative_loc[2]*img_size[0])
    ori_h = int(relative_loc[3]*img_size[1])
    return [ori_x, ori_y, ori_w, ori_h]


def iou_train(boxes1, boxes2, confidence_mask):

    """
    :param boxes1: masked-tensor: [batch size, cell size, cell size, boxes per cell, 4] => (relative_x_center, relative_y_center, relative_w, relative_h)
    :param boxes2: masked-tensor: [batch size, cell size, cell size, boxes per cell, 4] => (relative_x_center, relative_y_center, relative_w, relative_h)
    :return: tensor: [batch size, cell size, cell size, boxes per cell, 1] => (iou)
    """

    boxes_grid_x_offset = tf.tile(tf.reshape(tf.range(0, img_size[1], img_size[1]/cell_size), [1, cell_size]), [cell_size, 1])
    boxes_grid_y_offset = confidence_mask * tf.tile(tf.reshape(tf.transpose(boxes_grid_x_offset), [1, cell_size, cell_size, 1]), [tf.shape(boxes1)[0], 1, 1, 1])
    boxes_grid_x_offset = confidence_mask * tf.tile(tf.reshape(boxes_grid_x_offset, [1, cell_size, cell_size, 1]), [tf.shape(boxes1)[0], 1, 1, 1])
    boxes_grid_x_offset = tf.reshape(boxes_grid_x_offset, [tf.shape(boxes1)[0], cell_size, cell_size, 1])
    boxes_grid_y_offset = tf.reshape(boxes_grid_y_offset, [tf.shape(boxes1)[0], cell_size, cell_size, 1])

    boxes1_unstack = tf.unstack(boxes1, axis=-1)
    x_reverse_scale = (img_size[1]/cell_size) * boxes1_unstack[0] + boxes_grid_x_offset
    y_reverse_scale = (img_size[0]/cell_size) * boxes1_unstack[1] + boxes_grid_y_offset
    w_reverse_scale = img_size[1] * boxes1_unstack[2]
    h_reverse_scale = img_size[0] * boxes1_unstack[3]
    boxes1_reverse_scale = tf.stack([x_reverse_scale, y_reverse_scale, w_reverse_scale, h_reverse_scale], axis=-1)

    boxes2_unstack = tf.unstack(boxes2, axis=-1)
    x2_reverse_scale = (img_size[1]/cell_size) * boxes2_unstack[0] + boxes_grid_x_offset
    y2_reverse_scale = (img_size[0]/cell_size) * boxes2_unstack[1] + boxes_grid_y_offset
    w2_reverse_scale = img_size[1] * boxes2_unstack[2]
    h2_reverse_scale = img_size[0] * boxes2_unstack[3]
    boxes2_reverse_scale = tf.stack([x2_reverse_scale, y2_reverse_scale, w2_reverse_scale, h2_reverse_scale], axis=-1)

    boxes1_coord = tf.stack([boxes1_reverse_scale[..., 0] - boxes1_reverse_scale[..., 2]//2,
                             boxes1_reverse_scale[..., 1] - boxes1_reverse_scale[..., 3]//2,
                             boxes1_reverse_scale[..., 0] + boxes1_reverse_scale[..., 2]//2,
                             boxes1_reverse_scale[..., 1] + boxes1_reverse_scale[..., 3]//2], axis=-1)  # (x1, y1, x2, y2):left top, right down
    boxes2_coord = tf.stack([boxes2_reverse_scale[..., 0] - boxes2_reverse_scale[..., 2]//2,
                             boxes2_reverse_scale[..., 1] - boxes2_reverse_scale[..., 3]//2,
                             boxes2_reverse_scale[..., 0] + boxes2_reverse_scale[..., 2]//2,
                             boxes2_reverse_scale[..., 1] + boxes2_reverse_scale[..., 3]//2], axis=-1)  # (x1, y1, x2, y2):left top, right down

    left_top = tf.maximum(boxes1_coord[..., :2], boxes2_coord[..., :2])
    right_down = tf.minimum(boxes1_coord[..., 2:], boxes2_coord[..., 2:])

    intersect_length = tf.maximum(0.0, right_down-left_top)
    intersect_area = tf.reshape(intersect_length[..., 0] * intersect_length[..., 1], [tf.shape(intersect_length)[0], cell_size, cell_size, 1])

    boxes1_area = boxes1_reverse_scale[..., 2] * boxes1_reverse_scale[..., 3]
    boxes2_area = boxes2_reverse_scale[..., 2] * boxes2_reverse_scale[..., 3]
    union_area = tf.maximum(boxes1_area + boxes2_area - intersect_area, 1e-10)
    return tf.clip_by_value(intersect_area/union_area, 0.0, 1.0)


def reverse_scale(boxes):
    boxes_grid_x_offset = tf.tile(tf.reshape(tf.range(0, img_size[1], img_size[1] / cell_size), [1, cell_size]),
                                  [cell_size, 1])
    boxes_grid_y_offset = tf.tile(
        tf.reshape(tf.transpose(boxes_grid_x_offset), [1, cell_size, cell_size, 1]), [tf.shape(boxes)[0], 1, 1, 1])
    boxes_grid_x_offset = tf.tile(tf.reshape(boxes_grid_x_offset, [1, cell_size, cell_size, 1]),
                                                    [tf.shape(boxes)[0], 1, 1, 1])
    boxes_grid_x_offset = tf.reshape(boxes_grid_x_offset, [tf.shape(boxes)[0], cell_size, cell_size])
    boxes_grid_y_offset = tf.reshape(boxes_grid_y_offset, [tf.shape(boxes)[0], cell_size, cell_size])
    boxes1_unstack = tf.unstack(boxes, axis=-1)
    x_reverse_scale = (img_size[1] / cell_size) * tf.sigmoid(boxes1_unstack[0]) + boxes_grid_x_offset
    y_reverse_scale = (img_size[0] / cell_size) * tf.sigmoid(boxes1_unstack[1]) + boxes_grid_y_offset
    w_reverse_scale = img_size[1] * boxes1_unstack[2]
    h_reverse_scale = img_size[0] * boxes1_unstack[3]
    boxes_reverse_scale = tf.stack([x_reverse_scale, y_reverse_scale, w_reverse_scale, h_reverse_scale], axis=-1)

    return boxes_reverse_scale


def compute_cost(Z_out, Y, lambda_coord, lambda_noobj):

    predict_confidence = tf.reshape(Z_out[:, :cell_size*cell_size], [tf.shape(Z_out)[0], cell_size, cell_size, boxes_per_cell])

    predict_boxes = tf.reshape(Z_out[:, cell_size*cell_size:5*cell_size*cell_size], [tf.shape(Z_out)[0], cell_size, cell_size, boxes_per_cell, 4])
    predict_boxes_unstack = tf.unstack(predict_boxes, axis=-1)
    x_sigmoid = tf.sigmoid(predict_boxes_unstack[0])
    y_sigmoid = tf.sigmoid(predict_boxes_unstack[1])
    '''
    w_sqrt = tf.sqrt(tf.clip_by_value(predict_boxes_unstack[2], 0, math.inf))    # prevent negative number
    h_sqrt = tf.sqrt(tf.clip_by_value(predict_boxes_unstack[3], 0, math.inf))
    predict_boxes_coord_loss = tf.stack([x_sigmoid, y_sigmoid, w_sqrt, h_sqrt], axis=-1)
    predict_boxes_confidence_loss = tf.stack([x_sigmoid, y_sigmoid, tf.square(w_sqrt), tf.square(h_sqrt)], axis=-1)     # square the root of w and h, in order to be consistent with inference step
    '''
    predict_boxes_coord_loss = tf.stack([x_sigmoid, y_sigmoid, predict_boxes_unstack[2]/224, predict_boxes_unstack[3]/224], axis=-1)
    predict_boxes_confidence_loss = tf.stack([x_sigmoid, y_sigmoid, predict_boxes_unstack[2]/224, predict_boxes_unstack[3]/224], axis=-1)

    predict_classes = tf.reshape(Z_out[:, 5*cell_size*cell_size:], [tf.shape(Z_out)[0], cell_size, cell_size, num_class])

    confidence = tf.reshape(Y[..., 0], [tf.shape(Y)[0], cell_size, cell_size, 1])
    boxes = tf.reshape(Y[..., 1:5], [tf.shape(Y)[0], cell_size, cell_size, 1, 4])
    boxes = tf.tile(boxes, [1, 1, 1, boxes_per_cell, 1])  # duplicate true coordinate for every bounding box
    '''
    boxes_unstack = tf.unstack(boxes, axis=-1)
    boxes_sqrt_w = tf.sqrt(boxes_unstack[2])
    boxes_sqrt_h = tf.sqrt(boxes_unstack[3])
    boxes_sqrt_wh = tf.stack([boxes_unstack[0], boxes_unstack[1], boxes_sqrt_w, boxes_sqrt_h], axis=-1)
    '''
    classes = Y[..., 5:]


    # coordinate loss
    confidence_mask_coord = tf.tile(confidence, [1, 1, 1, 4])   # used when only one bounding box, not suitable for more than one
    confidence_mask_coord = tf.reshape(confidence_mask_coord, [tf.shape(confidence_mask_coord)[0], cell_size, cell_size, boxes_per_cell, 4])
    #coord_loss = lambda_coord * tf.reduce_mean(tf.square(confidence_mask_coord * predict_boxes_coord_loss - boxes_sqrt_wh))
    coord_loss = lambda_coord * tf.reduce_mean(tf.square(confidence_mask_coord * predict_boxes_coord_loss - boxes))

    # classes loss
    confidence_mask_for_classes = tf.tile(confidence, [1, 1, 1, num_class])   # used when only one bounding box, not suitable for more than one
    classes_loss = tf.reduce_mean(tf.square(confidence_mask_for_classes*predict_classes-classes))

    #confidence loss
    ground_truth_iou = iou_train(predict_boxes_confidence_loss, boxes, confidence)
    noobj_mask = tf.equal(confidence, 0)
    obj_mask = tf.equal(confidence, 1.0)
    obj_confidence_loss = tf.reduce_sum(tf.square(tf.boolean_mask(predict_confidence-confidence, obj_mask)))
    noobj_confidence_loss = lambda_noobj * tf.reduce_sum(tf.square(tf.boolean_mask(predict_confidence, noobj_mask)))
    confidence_loss = obj_confidence_loss + noobj_confidence_loss

    cost = coord_loss + classes_loss + confidence_loss
    return cost


def random_mini_batches(X, Y, minibatch_size=64, seed=0):

    n_sample = X.shape[0]
    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(n_sample))
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]

    num_batches = int(n_sample/minibatch_size)

    for n in range(num_batches):
        mini_batch_x = shuffled_X[n * minibatch_size:(n+1) * minibatch_size]
        mini_batch_y = shuffled_Y[n * minibatch_size:(n+1) * minibatch_size]
        mini_batches.append((mini_batch_x, mini_batch_y))

    if n_sample%minibatch_size != 0:
        mini_batch_x = shuffled_X[num_batches * minibatch_size:]
        mini_batch_y = shuffled_Y[num_batches * minibatch_size:]
        mini_batches.append((mini_batch_x, mini_batch_y))

    return mini_batches


def filter_boxes(confidence, boxes, classes, threshold=.6):
    box_scores = tf.tile(tf.reshape(confidence, [tf.shape(confidence)[0], cell_size, cell_size, 1]), [1, 1, 1,num_class]) * classes
    box_class = tf.argmax(box_scores, axis=-1)
    box_class_score = tf.reduce_max(box_scores, axis=-1)
    filter_mask = box_class_score > threshold
    confidence = tf.boolean_mask(box_class_score, filter_mask)
    coordinate = tf.boolean_mask(boxes, filter_mask)
    classes = tf.boolean_mask(box_class, filter_mask)
    return confidence, coordinate, classes


def yolo_v1_model(train_x, train_y, test_x, learning_rate=0.001, num_epoch=100, minibatch_size=64, dropout=0., regulation=0.):

    (num_sample, in_h, in_w, in_c) = train_x.shape
    (on_h, on_w, on_c) = train_y[0].shape
    costs = []
    lambda_coordinate = 5.0
    lambda_noobject = 0.5

    X, Y = create_placeholder(in_h, in_w, in_c, on_h, on_w, on_c)
    is_training = tf.placeholder(tf.bool, name='training_status')
    parameters = initialize_parameter()

    Z_out = forward_propagation(X, parameters, is_training, dropout)
    cost = compute_cost(Z_out, Y, lambda_coordinate, lambda_noobject)


    predict_Zout = tf.reshape(Z_out, [tf.shape(Z_out)[0], cell_size, cell_size, 15])
    reversed_coordinate = reverse_scale(predict_Zout[..., 1:5])
    filtered_confidence, filtered_coordinate, filtered_class = filter_boxes(predict_Zout[..., 0], reversed_coordinate, predict_Zout[..., 5:])
    selected_indices = tf.image.non_max_suppression(filtered_coordinate, filtered_confidence, 4, 0.6)
    selected_boxes = tf.gather(filtered_coordinate, selected_indices)
    selected_classes = tf.gather(filtered_class, selected_indices)

    confidence_mask_coord = tf.tile(tf.reshape(Y[..., 0], [tf.shape(Y)[0], cell_size, cell_size, 1]), [1, 1, 1, 4])   # used when only one bounding box, not suitable for more than one
    current_coord = Y[..., :3]
    if regulation > 0.:
        reg= 0
        for key in parameters:
            reg += tf.reduce_sum(tf.square(parameters[key]))
        cost += regulation * reg
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    saver = tf.train.Saver(max_to_keep=5)
    init = tf.global_variables_initializer()

    with tf.Session().as_default() as sess:

        sess.run(init)
        for epoch in range(num_epoch):
            minibatches = random_mini_batches(train_x, train_y, minibatch_size)
            num_minibatches = len(minibatches)
            epoch_cost = 0
            for batch in minibatches:
                (mini_x, mini_y) = batch
                _, mini_cost = sess.run([optimizer, cost], {X:mini_x, Y:mini_y, is_training:True})
                epoch_cost += mini_cost/num_minibatches
            if (epoch+1) % 1 == 0:
                print('loss epoch-{0}:{1}'.format(epoch+1, epoch_cost))
            if (epoch+1) % 50 == 0:
                saver.save(sess, 'model_file/yolo_model', global_step=epoch+1)
        #
        predict_boxes = sess.run(selected_boxes, {X: test_x, is_training: False})
        predict_classes = sess.run(selected_classes, {X: test_x, is_training: False})
        print(predict_boxes)
        print(predict_classes)
        mask = sess.run(confidence_mask_coord, {Y:train_y})
        coor = sess.run(current_coord, {Y:train_y})
    return parameters




if __name__ == '__main__':
    opt = argv[1]
    train_data_dir = argv[2]
    label_file = argv[3]
    test_dir = argv[4]
    if opt == '-tr':
        train_x, train_y = modify_data(train_data_dir, label_file)
        train_x = train_x/255
        test_x = read_img(test_dir)
        test_x = test_x/255
        learned_parameter = yolo_v1_model(train_x, train_y, test_x, minibatch_size=34, num_epoch=400)
