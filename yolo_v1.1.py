import tensorflow as tf
import numpy as np
import cv2
from sys import argv
import os
from collections import defaultdict


class_num = 10
output_grid_num = 7
img_size = {'w': 224, 'h': 224}


def read_img(data_dir):
    img_data = {}
    for img_file in os.listdir(data_dir):
        current_img_data = cv2.imread(os.path.join(data_dir, img_file), cv2.IMREAD_GRAYSCALE)
        current_img_data = np.asarray(current_img_data)
        current_img_data = current_img_data.reshape(current_img_data.shape[0], current_img_data.shape[1], 1)
        img_data[img_file[:5]] = current_img_data
    return img_data


def read_img_test(data_dir):
    img_data = []
    img_name = []
    for img_file in os.listdir(data_dir):
        current_img_data = cv2.imread(os.path.join(data_dir, img_file), cv2.IMREAD_GRAYSCALE)
        current_img_data = np.asarray(current_img_data)
        current_img_data = current_img_data.reshape(current_img_data.shape[0], current_img_data.shape[1], 1)
        img_data.append(current_img_data)
        img_name.append(img_file[:5])
    return img_name, np.asarray(img_data)


def read_label(file_dir):
    label_by_imgname = defaultdict(list)
    with open(file_dir, 'r') as input_file:
        for line in input_file.readlines():
            line_data = line.strip().split()
            label_by_imgname[line_data[0]].append(list(map(float, line_data[1:]))) # format: [confidence, x, y, w, h, classes]
    return label_by_imgname


def raw_to_train(img_dir, lable_dir):
    img_raw_data = read_img(img_dir)
    label_raw_data = read_label(lable_dir)

    grid_height = img_size['h']//output_grid_num
    grid_width = img_size['w']//output_grid_num
    X = []
    Y = []
    for img_id in img_raw_data:
        x = img_raw_data[img_id]
        y = np.zeros((output_grid_num, output_grid_num, 5 + class_num))
        raw_labels = label_raw_data[img_id]
        for each_label in raw_labels:
            confidence = each_label[0]
            ori_x_coord = each_label[1]
            ori_y_coord = each_label[2]
            grid_x = ori_x_coord//grid_width if ori_x_coord % grid_width != 0 else (ori_x_coord//grid_width) - 1 # assign the data to the grid
            grid_y = ori_y_coord//grid_height if ori_y_coord % grid_height != 0 else (ori_y_coord//grid_height) - 1
            relative_x_coordinate = (ori_x_coord % grid_width) / grid_width if ori_x_coord % grid_width != 0 else 1
            relative_y_coordinate = (ori_y_coord % grid_height) / grid_height if ori_y_coord % grid_height != 0 else 1
            relative_w = each_label[3]/img_size['w']
            relative_h = each_label[4]/img_size['h']
            y[int(grid_y), int(grid_x), :] = [confidence, relative_x_coordinate, relative_y_coordinate, relative_w,
                                              relative_h, *each_label[5:]]
        X.append(x)
        Y.append(y)
    return np.asarray(X), np.asarray(Y)


def create_placeholders(in_h, in_w, in_c, out_h, out_w, out_c):
    X = tf.placeholder(tf.float32, [None, in_h, in_w, in_c])
    Y = tf.placeholder(tf.float32, [None, out_h, out_w, out_c])
    return X, Y


def initialize_parameter():
    w1 = tf.get_variable('w1', [7, 7, 1, 32], initializer=tf.keras.initializers.he_normal())
    w2 = tf.get_variable('w2', [3, 3, 32, 64], initializer=tf.keras.initializers.he_normal())
    w3 = tf.get_variable('w3', [1, 1, 64, 16], initializer=tf.keras.initializers.he_normal())
    w4 = tf.get_variable('w4', [3, 3, 16, 32], initializer=tf.keras.initializers.he_normal())
    parameters = {'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4}
    return parameters


def forward_propagation(x, parameters, is_training):
    w1 = parameters['w1']
    w2 = parameters['w2']
    w3 = parameters['w3']
    w4 = parameters['w4']

    Z1 = tf.nn.conv2d(x, w1, [1,2,2,1], padding='SAME')  # 112
    Z1 = tf.layers.batch_normalization(Z1, training=is_training)
    A1 = tf.nn.leaky_relu(Z1, alpha=0.1)

    Z2 = tf.nn.conv2d(A1, w2, [1,1,1,1], padding='SAME')
    Z2 = tf.layers.batch_normalization(Z2, training=is_training)
    A2 = tf.nn.leaky_relu(Z2, alpha=0.1)

    P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  # 56

    Z3 = tf.nn.conv2d(P2, w3, [1,2,2,1], padding='SAME')  # 28
    Z3 = tf.layers.batch_normalization(Z3, training=is_training)
    A3 = tf.nn.leaky_relu(Z3, alpha=0.1)

    Z4 = tf.nn.conv2d(A3, w4, [1,2,2,1], padding='SAME')  # 14
    Z4 = tf.layers.batch_normalization(Z4, training=is_training)
    A4 = tf.nn.leaky_relu(Z4, alpha=0.1)

    P4 = tf.nn.max_pool(A4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  # 7

    conv_out = tf.layers.flatten(P4)

    fc_layer = tf.keras.layers.Dense(735, activation=None)
    fc_out = fc_layer(conv_out)
    fc_out = tf.reshape(fc_out, [tf.shape(x)[0], 7, 7, 15])
    return fc_out


def get_four_coord_original_scale(scaled_coord, true_mask):
    scaled_x = scaled_coord[..., 0]
    scaled_y = scaled_coord[..., 1]
    scaled_w = scaled_coord[..., 2]
    scaled_h = scaled_coord[..., 3]

    reversed_w = scaled_w * img_size['w']
    reversed_h = scaled_h * img_size['h']
    reversed_x_in_grid = scaled_x * img_size['w']/output_grid_num
    reversed_y_in_grid = scaled_y * img_size['h']/output_grid_num
    x_offset = tf.reshape(tf.tile(np.arange(0, output_grid_num), [output_grid_num]), [output_grid_num, output_grid_num]) * 32
    y_offset = tf.transpose(x_offset)

    reversed_x = (reversed_x_in_grid + tf.cast(x_offset, tf.float32))
    reversed_y = (reversed_y_in_grid + tf.cast(y_offset, tf.float32))

    x_min = (reversed_x - reversed_w/2) * true_mask
    x_max = (reversed_x + reversed_w/2) * true_mask
    y_min = (reversed_y - reversed_h/2) * true_mask
    y_max = (reversed_y + reversed_h/2) * true_mask

    four_coordinate = tf.stack([x_min, x_max, y_min, y_max], axis=-1)

    return four_coordinate


def iou_train(Z_out, Y):
    Z_coord = get_four_coord_original_scale(Z_out[..., 1:5], Y[..., 0])
    Y_coord = get_four_coord_original_scale(Y[..., 1:5], Y[..., 0])

    x_min = tf.maximum(Z_coord[..., 0], Y_coord[..., 0])
    x_max = tf.minimum(Z_coord[..., 1], Y_coord[..., 1])
    y_min = tf.maximum(Z_coord[..., 2], Y_coord[..., 2])
    y_max = tf.minimum(Z_coord[..., 3], Y_coord[..., 3])

    intersect_area = tf.maximum(0.0, (x_max-x_min) * (y_max-y_min))
    Z_area = (Z_coord[..., 1] - Z_coord[..., 0]) * (Z_coord[..., 3] - Z_coord[..., 2])
    Y_area = (Y_coord[..., 1] - Y_coord[..., 0]) * (Y_coord[..., 3] - Y_coord[..., 2])
    iou = intersect_area/(Z_area + Y_area - intersect_area)
    iou = tf.where(tf.is_nan(iou), tf.zeros_like(iou), iou)

    return iou


def compute_cost(Z_out, Y, lambda_coord, lambda_noobj):
    # coordinate loss
    true_confidence_mask = Y[..., 0]
    predicted_x_masked = true_confidence_mask * Z_out[..., 1]
    predicted_y_masked = true_confidence_mask * Z_out[..., 2]
    coordinate_x_loss = tf.reduce_sum(tf.square(predicted_x_masked-Y[..., 1]))
    coordinate_y_loss = tf.reduce_sum(tf.square(predicted_y_masked-Y[..., 2]))
    coordinate_loss = lambda_coord * (coordinate_x_loss + coordinate_y_loss)

    # size loss
    predicted_w_masked_sqrt = tf.sqrt(true_confidence_mask * Z_out[..., 3])
    predicted_w_masked_without_nan = tf.where(tf.is_nan(predicted_w_masked_sqrt), tf.zeros_like(predicted_w_masked_sqrt),
                                              predicted_w_masked_sqrt)
    predicted_h_masked_sqrt = tf.sqrt(true_confidence_mask * Z_out[..., 4])
    predicted_h_masked_without_nan = tf.where(tf.is_nan(predicted_h_masked_sqrt), tf.zeros_like(predicted_h_masked_sqrt),
                                              predicted_h_masked_sqrt)

    w_sqrt_loss = tf.reduce_sum(tf.square(predicted_w_masked_without_nan - tf.sqrt(Y[..., 3])))
    h_sqrt_loss = tf.reduce_sum(tf.square(predicted_h_masked_without_nan - tf.sqrt(Y[..., 4])))
    size_loss = lambda_coord * (w_sqrt_loss + h_sqrt_loss)

    # classes loss
    mask_shape = tf.shape(true_confidence_mask)
    true_confidence_mask_for_class = tf.reshape(true_confidence_mask, [mask_shape[0], mask_shape[1], mask_shape[2], 1])
    predicted_class_masked = true_confidence_mask_for_class * Z_out[..., 5:]
    class_loss = tf.reduce_sum(tf.square(predicted_class_masked-Y[..., 5:]))

    # confidence loss with object
    iou = iou_train(Z_out, Y)
    confidence = iou * Z_out[..., 0]
    confidence_loss_object_with_nans = tf.square(Y[..., 0] - confidence)
    confidence_loss_object_without_nans = tf.where(tf.is_nan(confidence_loss_object_with_nans),      # nan value would cause the sum function to return nan
                                                   tf.zeros_like(confidence_loss_object_with_nans),
                                                   confidence_loss_object_with_nans)
    confidence_loss_object_without_nans = tf.reduce_sum(confidence_loss_object_without_nans)

    # confidence loss without object
    noobj_mask = tf.ones_like(true_confidence_mask)-true_confidence_mask
    confidence_loss_noobj = lambda_noobj * tf.reduce_sum(tf.square(noobj_mask * Z_out[..., 0]))

    total_loss = coordinate_loss + class_loss + size_loss + confidence_loss_object_without_nans + confidence_loss_noobj

    return total_loss


def compute_simple_cost(Z_out, Y, lambda_obj, lambda_noobj):
    true_mask = Y[..., 0]
    true_mask_shape = tf.shape(true_mask)
    true_mask = tf.reshape(true_mask, [true_mask_shape[0], true_mask_shape[1], true_mask_shape[2], 1])
    nobj_mask = 1 - true_mask

    # sqrt size loss
    Y_w = Y[..., 3]
    Z_out_w = Z_out[..., 3]
    w_diff_square = tf.reduce_sum(tf.square(Y_w-Z_out_w) * tf.squeeze(true_mask))
    Y_h = Y[..., 4]
    Z_out_h = Z_out[..., 4]
    h_diff_square = tf.reduce_sum(tf.square(Y_h-Z_out_h) * tf.squeeze(true_mask))
    # get the distance between true and predicted result, eliminate the effect causing by negative value
    Z_w_modifiy_negative = tf.where(Z_out_w < 0, Y_w-Z_out_w, Z_out_w)
    Z_h_modifiy_negative = tf.where(Z_out_h < 0, Y_h-Z_out_h, Z_out_h)
    size_loss = tf.reduce_sum(tf.square(tf.sqrt(Y_w)-tf.sqrt(tf.abs(Z_w_modifiy_negative))) +
                              tf.square(tf.sqrt(Y_h)-tf.sqrt(tf.abs(Z_h_modifiy_negative))))

    loss_all = tf.square(Y - Z_out)
    loss_with_obj = lambda_obj * (tf.reduce_sum(loss_all * true_mask) - w_diff_square - h_diff_square + size_loss)
    loss_without_obj = lambda_noobj * tf.reduce_sum(loss_all[..., 0] * tf.squeeze(nobj_mask))

    return loss_with_obj + loss_without_obj


def random_mini_batches(X, Y, minibatch_size=64, seed=1):

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


def filter_confidence(predict_confidence, threshold=.5):
    filter_mask = predict_confidence > threshold
    return predict_confidence * filter_mask, filter_mask


def transform_predict_result(predict_result):
    filtered_confidence, mask = filter_confidence(predict_result[..., 0])
    indices = tf.where(mask)
    extracted_output = tf.gather_nd(predict_result, indices)
    x = (extracted_output[..., 1] + tf.cast(indices[..., 1], tf.float32)) * img_size['w']/output_grid_num
    y = (extracted_output[..., 2] + tf.cast(indices[..., 0], tf.float32)) * img_size['h']/output_grid_num
    w = extracted_output[..., 3] * img_size['w']
    h = extracted_output[..., 4] * img_size['h']

    predict_class = tf.cast(tf.argmax(extracted_output[..., 5:], axis=-1), tf.float32)
    transformed_result = tf.stack([extracted_output[..., 0], x, y, w, h, predict_class], axis=-1)
    return transformed_result


def draw_bouning_boxes(img, label_info):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for each in label_info:
        (confidence, x, y, w, h, label) = each
        x_min = int(x - w/2)
        x_max = int(x + w/2)
        y_min = int(y - h/2)
        y_max = int(y + h/2)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        cv2.putText(img, '{0}%:{1}'.format(confidence*100, label), (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 0), 1)
    cv2.imwrite('test.jpg', img)
    return 0


def yolo_model(train_x, train_y, test_x, lr_rate=0.001, num_epoch=10, minibatch_size=32):
    (num_sample, in_h, in_w, in_c) = train_x.shape
    (out_h, out_w, out_c) = train_y[0].shape
    lambda_coordinate = 5.0
    lambda_noobject = 0.5

    X, Y = create_placeholders(in_h, in_w, in_c, out_h, out_w, out_c)
    parameters = initialize_parameter()
    is_training = tf.placeholder(tf.bool, name='is_training')

    forward_out = forward_propagation(X, parameters, is_training)
    #loss = compute_cost(forward_out, Y, lambda_coordinate, lambda_noobject)
    simple_loss = compute_simple_cost(forward_out, Y, lambda_coordinate, lambda_noobject)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_rate).minimize(simple_loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session().as_default() as sess:
        init.run()
        for epoch in range(num_epoch):
            minibatches = random_mini_batches(train_x, train_y, minibatch_size)
            num_minibatches = len(minibatches)
            epoch_cost = 0
            for batch in minibatches:
                (mini_x, mini_y) = batch
                _, mini_cost = sess.run([optimizer, simple_loss], {X: mini_x, Y: mini_y, is_training: True})
                epoch_cost += mini_cost/num_minibatches
            if (epoch + 1) % 1 == 0:
                 print('loss epoch-{0}:{1}'.format(epoch + 1, epoch_cost))
            if (epoch + 1) % 100 == 0:
                saver.save(sess, 'yolo_model', global_step=epoch+1)

        predict_result = sess.run(forward_out, {X: test_x, is_training: False})
        sess.run(transform_predict_result(predict_result))
    return parameters


if __name__ == '__main__':
    train_data_dir = argv[1]
    label_file_dir = argv[2]
    test_data_dir = argv[3]
    train_x, train_y = raw_to_train(train_data_dir, label_file_dir)
    test_id, test_x = read_img_test(test_data_dir)
    learned_parameters = yolo_model(train_x, train_y, test_x)
