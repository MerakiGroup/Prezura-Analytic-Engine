# This is a boilerplate of clustering a 1d array, when only one attribute
# is present in the data set (maybe a one pressure sensor). This is only
# for the demonstration purpose, about how we are planing to identify
# abnormalities of the data set since not enough data is not still present
# for the Prezura analytic, this will generate a given number of centroid
# in a 1d data set

from __future__ import division, print_function, unicode_literals
from tensorflow.contrib.learn.python.learn.estimators import kmeans
import tensorflow as tf


def input_fn_1d(input_1d):
    """
    Covert an numpy array to a tensorflow object

    :param input_1d the 1d data set
    """
    input_t = tf.convert_to_tensor(input_1d, dtype=tf.float32)
    input_t = tf.expand_dims(input_t, 1)
    return input_t, None


def generate_cluster(k, data_set):
    """
    This will generate centroids of k cluster from the given date set

    :param k number of clusters to generate
    :param data_set input data set
    """
    k_means_estimator = kmeans.KMeansClustering(num_clusters=k)
    k_means_estimator.fit(input_fn=lambda: input_fn_1d(data_set), steps=1000)
    return k_means_estimator.clusters()

