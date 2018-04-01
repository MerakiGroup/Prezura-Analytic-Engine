import sample_generator
import numpy as np

n_features = 2
n_clusters = 3
n_samples_per_cluster = 500
seed = 700
embiggen_factor = 70

np.random.seed(seed)

centroids, samples = sample_generator.create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)

print(centroids)
print(samples)