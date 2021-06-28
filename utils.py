import tensorflow as tf
import numpy as np

from os import listdir
from os.path import isfile, join

import xml.etree.ElementTree as ET

import xmltodict
from PIL import Image

import matplotlib.pyplot as plt
import cv2


def bb_intersection_over_union(pois: tf.Tensor, bboxgt: tf.Tensor) -> tf.Tensor:
    """this code calculates IOU for pois and ground truth bounding boxes, returns tensor of iou for every poi

    Args:
        pois (tf.Tensor): tensor of shape (k, 4), last dimension: [xmin, ymin, xmax, ymax] 
        bboxgt (tf.Tensor): tensor of shape (m, 4 ), last dimension: [xmin, ymin, xmax, ymax] 

    Returns:
        tf.Tensor: tensor of shape (k,)
    """

    # change dtypes

    pois = tf.cast(pois, dtype=tf.float32)
    bboxgt = tf.cast(bboxgt, dtype=tf.float32)

    number_of_bboxgt = tf.shape(bboxgt)[0]
    number_of_pois = tf.shape(pois)[0]

    expand_dims_pois = tf.repeat(tf.expand_dims(
        pois, axis=1), number_of_bboxgt, axis=1)
    expand_dims_bboxgt = tf.repeat(tf.expand_dims(
        bboxgt, axis=0), number_of_pois, axis=0)

    xA = tf.math.maximum(
        expand_dims_pois[:, :, 0], expand_dims_bboxgt[:, :, 0])
    yA = tf.math.maximum(
        expand_dims_pois[:, :, 1], expand_dims_bboxgt[:, :, 1])
    xB = tf.math.minimum(
        expand_dims_pois[:, :, 2], expand_dims_bboxgt[:, :, 2])
    yB = tf.math.minimum(
        expand_dims_pois[:, :, 3], expand_dims_bboxgt[:, :, 3])

    interArea = tf.math.maximum(tf.constant(0, dtype=tf.float32), (xB - xA)) * \
        tf.math.maximum(tf.constant(0, dtype=tf.float32), (yB - yA))

    boxAArea = (expand_dims_pois[:, :, 2] - expand_dims_pois[:, :, 0]) * \
        (expand_dims_pois[:, :, 3] - expand_dims_pois[:, :, 1])
    boxBArea = (expand_dims_bboxgt[:, :, 2] - expand_dims_bboxgt[:, :, 0]) * (
        expand_dims_bboxgt[:, :, 3] - expand_dims_bboxgt[:, :, 1])
    iou = interArea / (boxAArea + boxBArea - interArea)
    iou_reduced = tf.reduce_max(iou, axis=1)

    return iou_reduced

def to_box(x: tf.Tensor) -> tf.Tensor:  # tf compatible
    """converts from [x_center, y_center, width, height] to [xmin, ymin, xmax, ymax] 

    Args:
        x (tf.Tensor): tensor of shape (None, 4) where last dimension [x_center, y_center, width, height]

    Returns:
        tf.Tensor: tensor of shape (None, 4), where last dimension [xmin, ymin, xmax, ymax], dtype = tf.float32
    """
    xmin = (x[:, 0] - (x[:, 2]/2))
    xmax = (x[:, 0] + (x[:, 2]/2))
    ymin = (x[:, 1] - (x[:, 3]/2))
    ymax = (x[:, 1] + (x[:, 3]/2))
    res = tf.stack((xmin, ymin, xmax, ymax), axis=0)
    return tf.cast(tf.transpose(res), dtype=tf.float32)

def sampler(tensor: tf.Tensor, sample_size: int = 255) -> tf.Tensor:
    """this function will sample rows of tensor, if sample size is biger than number of rows in tensor, sample size will default to # of rowns in tensor

    Args:
        tensor (tf.Tensor): tensor 
        sample_size (int, optional): number of samples to generate (if possible). Defaults to 255.

    Returns:
        tf.Tensor: tensor of shape (sample_size, None, None)
    """
    if tf.shape(tensor)[0] < sample_size:
        sample_size = tf.shape(tensor)[0]
    #tf.print("shape A:", tf.shape(tensor))
    sample = tf.random.uniform_candidate_sampler(
        tensor, 1, sample_size, unique=True, range_max=tf.shape(tensor)[0], seed=None, name=None)[0]
    return(sample)

def generate_bbox_coords(scales: np.array = np.array([0.5, 1, 2]), dims: np.array = np.array([0.01, 0.05, 0.1]), n_points: int = 400) -> np.array:
    """ this function will generate array of bounding boxes dependant on scales and dims, 
    BE AWARE: scales simetric! [0.5, 1, 2] 

    n_points is number of points in image kernel, for example, image kernel is 20*20 *None , su number of points will be 400 

    Args:
        scales (np.array, optional): should be simetric. Defaults to np.array([0.5,1,2]).
        dims (np.array, optional): dimensions. Defaults to np.array([0.01, 0.05, 0.1]).
        n_points (int, optional): number of points in image kernel. Defaults to 400.


    Returns:
        np.array: shape (n_points * scales_shape * dims_shape, 4)  [x_center, y_center, width, height]
    """
    # generates bboxes along image

    x_bound = np.outer(scales.astype(float), dims.astype(float))
    y_bound = np.reshape(x_bound, (-1))
    k = y_bound.shape[0]
    x_bound = np.reshape(np.flipud(x_bound), (-1))
    col_stack = np.stack((x_bound, y_bound), axis=1)
    xy_bound = np.tile(col_stack, (n_points, 1))

    x = np.arange(start=1/40, stop=1 + (1/40), step=1/20)
    y = np.arange(start=1/40, stop=1 + (1/40), step=1/20)
    xx, yy = np.meshgrid(x, y)
    xc = np.repeat(np.expand_dims(xx, 2), repeats=k,  axis=2).reshape(-1)
    yc = np.repeat(np.expand_dims(yy, 2), repeats=k,  axis=2).reshape(-1)
    centers_xy = np.column_stack((xc, yc))

    return np.column_stack((centers_xy, xy_bound))

def positive_negative_sampler(iou: tf.Tensor, range_positive: float, range_negative: float) -> tuple:
    """returns indexes of pois with positive and negative values, alos weights for log loss 

    Args:
        iou (tf.Tensor): tensor of IOU values
        range_positive (float): values from which ious are positive
        range_negative (float): values from which ious are negative
    Returns:
        tuple: positive iou ids, negative iou ids, positive weight, negative weight
    """

    positive_idx = tf.where(iou > range_positive)
    if tf.shape(positive_idx)[0] == 0:
        positive_idx = tf.where(iou == tf.math.reduce_max(iou))

    # sample positives (for balancing purposes, still wont be balanced equally)
    positive_idx_sampled = tf.gather(positive_idx, sampler(positive_idx, 255))
    n_positives = tf.shape(positive_idx_sampled)[0]

    # get idx of negatives
    negatives_idx = tf.where(iou < range_negative)
    negative_idx_count = 255 + (255 - n_positives)
    if tf.shape(negatives_idx)[0] == 0:
        negatives_idx = tf.where(iou == tf.math.reduce_min(iou))

    negative_idx_sampled = tf.gather(
        negatives_idx, sampler(negatives_idx, negative_idx_count))
    n_negatives = tf.shape(negative_idx_sampled)[0]

    # classes will be not balanced so calculate class weights:
    # wj=n_samples / (n_classes * n_samplesj)
    n_classes = 2
    n_samples = n_negatives + n_positives
    w_positives = tf.cast(
        n_samples / (n_classes * n_positives), dtype=tf.float32)
    w_negatives = tf.cast(
        n_samples / (n_classes * n_negatives), dtype=tf.float32)

    #tf.print(w_positives, w_negatives)

    return (positive_idx_sampled, negative_idx_sampled, w_positives, w_negatives)

def positive_negative_sampler_topk(iou, range_positive_topk, range_negative):

    positive_idx = tf.math.top_k(iou, 10)[1]
    negatives_idx = tf.where(iou < range_negative)
    negative_idx_sampled = tf.py_function(
        func=sampler, inp=[negatives_idx, 10], Tout=[tf.int64], name="sampler")
    negative_idx_sampled = tf.gather(negatives_idx, negative_idx_sampled)
    return (positive_idx, negative_idx_sampled)

def loglos(positive_scores: tf.Tensor, negative_scores: tf.Tensor, w_pos: float = 1, w_neg: float = 1) -> tf.Tensor:
    """logisitc loss

    Args:
        positive_scores (tf.Tensor): scores of positive ious
        negative_scores (tf.Tensor): scores of negative ious
        w_pos (float, optional): weights for positive ious. Defaults to 1.
        w_neg (float, optional): weights for negative ious. Defaults to 1.

    Returns:
        tf.Tensor: class loss 
    """
    negative_loss = tf.math.reduce_mean(
        w_neg * tf.math.log((1-negative_scores) + (10**(-20))))
    positive_loss = tf.math.reduce_mean(
        w_pos * tf.math.log(positive_scores + (10**(-20))))
    class_loss = -(positive_loss + negative_loss)

    return class_loss

def xyxy_to_xywh(y_true: tf.Tensor) -> tf.Tensor:
    """converts tensor of shape (None, 4) where last dimension is [xmin,ymin,xmax,ymax] to tensor of same shae where last dimension is [x_center, y_center, width, height]

    Args:
        y_true (tf.Tensor): tensor of shape (None, 4) where last dimension is [xmin,ymin,xmax,ymax]

    Returns:
        tf.Tensor: tensor of shape (None, 4) where last dimension is [x_center, y_center, width, height]
    """
    y_true = tf.cast(y_true, dtype=tf.float32)
    h_true = y_true[:, 3] - y_true[:, 1]
    w_true = y_true[:, 2] - y_true[:, 0]
    x_center_true = y_true[:, 0] + (w_true/2)
    y_center_true = y_true[:, 1] + (h_true/2)
    true_cords = tf.stack(
        [x_center_true, y_center_true, w_true, h_true], axis=1)
    return true_cords

def l1_loss(true_cords: tf.Tensor, bbox_positive: tf.Tensor, delta: tf.Tensor) -> tf.Tensor:
    """calculates l1 loss for predictions

    Args:
        true_cords (tf.Tensor): cordinates of ground truth bboxes
        bbox_positive (tf.Tensor): cordinates of positive bboxes
        delta (tf.Tensor): predicted differences of bbox cordinates

    Returns:
        tf.Tensor: l1 loss
    """

    # cast to floats
    bbox_positive = tf.cast(bbox_positive, tf.float32)
    true_cords = tf.cast(true_cords, tf.float32)
    delta = tf.cast(delta, tf.float32)

    adjusted_bbox = bbox_positive + delta

    adjusted_bbox_dim0 = tf.shape(adjusted_bbox)[0]
    true_cords_dim0 = tf.shape(true_cords)[0]

    expand_dims_bbox_positive = tf.repeat(
        adjusted_bbox, true_cords_dim0, axis=1)
    expand_dims_true_cords = tf.repeat(tf.expand_dims(
        true_cords, axis=0), adjusted_bbox_dim0, axis=0)

    difference = expand_dims_true_cords - expand_dims_bbox_positive
    abs_diference = tf.math.abs(difference, name="abs_diference")
    sum_coord_difference = tf.math.reduce_sum(
        abs_diference, axis=2, name="sum_coord_difference")
    min_sum = tf.math.reduce_min(sum_coord_difference, axis=1, name="min_sum")
    l1_loss = tf.math.reduce_mean(min_sum, name="l1_loss")

    return l1_loss

def my_loss_fn(y_true, output_scores, output_deltas, bboxes, lambda_coef=1, range_positive=0.7, range_negative=0.1):

    y_true = tf.squeeze(y_true.to_tensor(), axis=0)
    deltas = tf.reshape(output_deltas, (-1, 4))
    scores = tf.reshape(output_scores, (-1, 1))

    # map_fn is very slow
    pois = to_box(bboxes)

    iou = bb_intersection_over_union(pois, y_true)

    positive_idx_sampled, negative_idx_sampled, w_positives, w_negatives = tf.py_function(func=positive_negative_sampler,
                                                                                          inp=[
                                                                                              iou, range_positive, range_negative],
                                                                                          Tout=[
                                                                                              tf.int64, tf.int64, tf.float32, tf.float32],
                                                                                          name="sampler")

    negative_scores = tf.gather_nd(scores, negative_idx_sampled)
    positive_scores = tf.gather_nd(scores, positive_idx_sampled)

    class_loss = tf.py_function(func=loglos, inp=[
                                positive_scores, negative_scores, w_positives, w_negatives], Tout=[tf.float32], name="log_loss")

    # get positive deltas and bbox
    bbox_positive = tf.gather(bboxes, positive_idx_sampled)
    deltas_positive = tf.gather(deltas, positive_idx_sampled)

    # transform true coords from xmin,ymin,xmax,ymax to x,y,w,h format
    true_cords = xyxy_to_xywh(y_true)

    # calculate l1 loss
    #regression_loss_l1 = l1_loss(true_cords, bbox_positive, deltas_positive)

    stack = tf.concat((bbox_positive, deltas_positive), axis = 1)

    loss_l1_vector = tf.vectorized_map(fn = lambda x: loss_l1_vectorized(x, true_cords), elems = stack)
    regression_loss_l1 = tf.math.reduce_mean(loss_l1_vector)

    # final loss
    loss = tf.cast(class_loss, dtype=tf.float32) + (tf.cast(lambda_coef,
                                                            dtype=tf.float32) * tf.cast(regression_loss_l1, dtype=tf.float32))

    return loss, class_loss, regression_loss_l1

def loss_l1_vectorized(stack: tf.Tensor, true_cords: tf.Tensor) -> tf.Tensor:
    bbox_positive = stack[0]
    positive_deltas = stack[1]
    difference_true = bbox_positive - true_cords
    center_distance = tf.reduce_sum(tf.abs(difference_true[:,0:2]), axis = 1)
    which_min = tf.where(center_distance == tf.reduce_min(center_distance))
    loss = tf.math.reduce_sum(tf.math.abs(tf.gather(difference_true, which_min) - positive_deltas))
    return loss 

def read_data(path: str):

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    images_data = list()
    bbox_data = list()
    for name in onlyfiles:
        with open(path + "\\" + name) as fd:
            doc = xmltodict.parse(fd.read())

        if type(doc["annotation"]["object"]) is list:
            bbox = np.array([list(i["bndbox"].values())
                            for i in doc["annotation"]["object"]], dtype=int)
        else:
            bbox = np.array(
                [list(doc["annotation"]["object"]["bndbox"].values())], dtype=int)
        image_path = doc["annotation"]["path"]
        image = Image.open(image_path)
        imagearr = tf.keras.preprocessing.image.img_to_array(image)/255
        # be aware of scaling
        bbox = bbox / np.tile(np.flip(imagearr.shape[:2]), 2)
        imagearr = tf.image.resize(imagearr, (640, 640))
        images_data.append(imagearr)
        bbox_data.append(bbox)

    bbox_data = tf.ragged.constant(bbox_data)
    images_data = tf.convert_to_tensor(images_data)

    return bbox_data, images_data

def plot_image(img, bbox):
    sized_bbox = tf.cast(tf.cast(tf.tile(
        img.shape[:2], [2]), dtype=tf.float32) * bbox.numpy(), dtype=tf.int32).numpy()
    image_with_bounding_box = img.numpy()
    for i in sized_bbox:
        image_with_bounding_box = cv2.rectangle(
            image_with_bounding_box, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 10)
    return plt.imshow(image_with_bounding_box)
