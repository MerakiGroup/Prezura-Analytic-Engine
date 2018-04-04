# This module contain a keman clustering on a multidimensional data space
# which need to be configured and validated after a accurate data validation
# model is found

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed):

    """
    Create a set of centroids and generate a sample dataset

    :param n_clusters Number of clusters need to be generated
    :param n_samples_per_cluster number of data points per cluster
    :param n_features number of features
    :param embiggen_factor embiggen factor of generating data
    :param seed Used to create a random seed for the distribution
    """
    np.random.seed(seed)
    slices = []
    centroids = []

    # Create samples for each cluster
    for i in range(n_clusters):

        # A tensor of the specified shape filled with random normal values.
        samples = tf.random_normal((n_samples_per_cluster, n_features),
                                   mean=0.0, stddev=5.0, dtype=tf.float32, seed=seed, name="cluster_{}".format(i))

        current_centroid = (np.random.random((1, n_features)) * embiggen_factor) - (embiggen_factor/2)
        centroids.append(current_centroid)
        samples += current_centroid
        slices.append(samples)

    # Create a big "samples" dataset
    samples = tf.concat(slices, 0, name='samples')
    centroids = tf.concat(centroids, 0, name='centroids')
    return centroids, samples


def plot_clusters(all_samples, centroids, n_samples_per_cluster):

    """
    Displays the plot visualisation

    :param all_samples: the data set
    :param centroids: generated centroids
    :param n_samples_per_cluster:  one cluster sample size
    :return:
    """

    # Plot out the different clusters
    # Choose a different colour for each cluster
    colour = plt.cm.rainbow(np.linspace(0, 1, len(centroids)))

    for i, centroid in enumerate(centroids):
        # Grab just the samples fpr the given cluster and plot them out with a new colour
        samples = all_samples[i * n_samples_per_cluster:(i + 1) * n_samples_per_cluster]
        plt.scatter(samples[:, 0], samples[:, 1], c=colour[i])
        # Also plot centroid
        plt.plot(centroid[0], centroid[1], markersize=35, marker="x", color='k', mew=10)
        plt.plot(centroid[0], centroid[1], markersize=30, marker="x", color='m', mew=5)
    plt.show()
