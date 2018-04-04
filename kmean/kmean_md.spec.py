import kmean_md
import numpy as np
import unittest
import tensorflow as tf

n_features = 2
n_clusters = 3
n_samples_per_cluster = 500
seed = 700
embiggen_factor = 70

np.random.seed(seed)

# Generating sample
centroids, samples = kmean_md.create_samples(
    n_clusters,
    n_samples_per_cluster,
    n_features,
    embiggen_factor,
    seed
)

model = tf.global_variables_initializer()

# Tensorflow Executing the kmean generator
with tf.Session() as session:
    sample_values = session.run(samples)
    centroid_values = session.run(centroids)
    session.close()

kmean_md.plot_clusters(sample_values, centroid_values, n_samples_per_cluster)


# Test Start
class TestStringMethods(unittest.TestCase):

    def setUp(self):
        print("In method", self._testMethodName)

    def test_centroid_validation(self):
        self.assertEqual(centroids.shape, (3, 2))

    def test_samples_validation(self):
        self.assertEqual(samples.shape, (1500, 2))


if __name__ == '__main__':
    unittest.main()

