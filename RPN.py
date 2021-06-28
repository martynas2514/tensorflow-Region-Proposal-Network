import tensorflow as tf
import numpy as np
from utils import my_loss_fn, generate_bbox_coords
import utils

loss_tracker = tf.keras.metrics.Mean(name="loss")
class_loss_tracker = tf.keras.metrics.Mean(name="class_loss")
l1_loss_tracker = tf.keras.metrics.Mean(name="l1_loss")

class RPN(tf.keras.Model):

    def __init__(self, k, lambda_coef=10,
                 range_positive=0.7,
                 range_negative=0.1,
                 scales=np.array([0.5, 1, 2]),
                 dims=np.array([0.01, 0.05, 0.1])):
        super(RPN, self).__init__()

        # coefs
        self.lambda_coef = lambda_coef
        self.range_positive = range_positive
        self.range_negative = range_negative
        k = scales.shape[0] * dims.shape[0]
        self.bboxes = generate_bbox_coords(scales, dims)

        # layers
        self.convolution_3x3 = tf.keras.layers.Conv2D(
            filters=512, padding="same", kernel_size=(3, 3), name="3x3")

        self.output_deltas_layer = tf.keras.layers.Conv2D(filters=4 * k, kernel_size=(
            1, 1), activation="linear", kernel_initializer="uniform", name="deltas1")

        self.output_scores_layer = tf.keras.layers.Conv2D(filters=k, kernel_size=(
            1, 1), activation="sigmoid", kernel_initializer="uniform", name="scores1")

        
        self.loss_fn = my_loss_fn

    def call(self, inputs):
        x = self.convolution_3x3(inputs)
        output_deltas = self.output_deltas_layer(x)
        output_scores = self.output_scores_layer(x)
        return [output_deltas, output_scores]

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y_true = data

        # because its a ragged tensor
        y_true = tf.squeeze(y_true.to_tensor(), axis=0)
        pois = utils.to_box(self.bboxes)
        iou = utils.bb_intersection_over_union(pois, y_true)

        positive_idx_sampled, negative_idx_sampled, w_positives, w_negatives = tf.py_function(func=utils.positive_negative_sampler,
                                                                                              inp=[
                                                                                                  iou, self.range_positive, self.range_negative],
                                                                                              Tout=[
                                                                                                  tf.int64, tf.int64, tf.float32, tf.float32],
                                                                                              name="sampler")

        bbox_positive = tf.cast(tf.gather(self.bboxes, positive_idx_sampled), dtype = tf.float32)

        true_cords = utils.xyxy_to_xywh(y_true)

        with tf.GradientTape() as tape:
            output_deltas, output_scores = self(x)  # Forward pass

            deltas = tf.reshape(output_deltas, (-1, 4))
            scores = tf.reshape(output_scores, (-1, 1))

            negative_scores = tf.gather_nd(scores, negative_idx_sampled)
            positive_scores = tf.gather_nd(scores, positive_idx_sampled)

            class_loss = tf.py_function(func=utils.loglos, inp=[
                                        positive_scores, negative_scores, w_positives, w_negatives], Tout=[tf.float32], name="log_loss")

            deltas_positive = tf.gather(deltas, positive_idx_sampled)

            stack = tf.concat((bbox_positive, deltas_positive), axis = 1)

            loss_l1_vector = tf.vectorized_map(fn = lambda x: utils.loss_l1_vectorized(x, true_cords), elems = stack)
            regression_loss_l1 = tf.math.reduce_mean(loss_l1_vector)

            # final loss
            loss = tf.cast(class_loss, dtype=tf.float32) + (tf.cast(self.lambda_coef,
                                                                    dtype=tf.float32) * tf.cast(regression_loss_l1, dtype=tf.float32))
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            #loss = self.my_loss_fn(y, output_scores, output_deltas, self.bboxes, lambda_coef = self.lambda_coef, range_positive =self.range_positive , range_negative = self.range_negative )
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Return a dict mapping metric names to current value

        # add losses as metrics 
        loss_tracker.update_state(loss)
        class_loss_tracker.update_state(class_loss)
        l1_loss_tracker.update_state(regression_loss_l1)

        return {"loss": loss_tracker.result(), "class_loss": class_loss_tracker.result(), "l1_loss": l1_loss_tracker.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker, class_loss_tracker, l1_loss_tracker]
