from __future__ import division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt

input_1d_x = np.array([
    1, 2, 3.0, 4, 5, 126, 21, 33, 6, 127, 66, 23, 110, 4, 8, 33, 102
])

def input_fn_1d(input_1d):
    input_t = tf.convert_to_tensor(input_1d, dtype=tf.float32)
    input_t = tf.expand_dims(input_t, 1)

    return input_t, None

plt.scatter(input_1d_x, np.zeros_like(input_1d_x), s=500)
plt.show()

from tensorflow.contrib.learn.python.learn.estimators import kmeans

k_means_estimator = kmeans.KMeansClustering(num_clusters=3)

fit = k_means_estimator.fit(input_fn=lambda: input_fn_1d(input_1d_x), steps=1000)

clusters_1d = k_means_estimator.clusters()

print(clusters_1d)

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(input_1d_x, np.zeros_like(input_1d_x),s=300, marker='o')
ax1.scatter(clusters_1d, np.zeros_like(clusters_1d), c='r', s=200, marker='s')

plt.show()

