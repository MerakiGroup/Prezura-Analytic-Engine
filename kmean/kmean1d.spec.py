import unittest
import kmean1d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

input_1d_x = np.array([
    1, 2, 5, 9, 8, 3, 6, 8,  51, 63, 55, 69, 64, 52, 57, 52, 101, 103, 109, 108, 106, 111, 110
])

# Show Current Data Set in plot scatter
plt.scatter(input_1d_x, np.zeros_like(input_1d_x), s=500)
plt.show()

num_of_clusters = 3

clusters_1d = kmean1d.generate_cluster(num_of_clusters, input_1d_x)

# Creating the plot with centroids
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(input_1d_x, np.zeros_like(input_1d_x), s=300, marker='o')
ax1.scatter(clusters_1d, np.zeros_like(clusters_1d), c='r', s=200, marker='s')

plt.show()


# Test Start
class TestStringMethods(unittest.TestCase):

    def setUp(self):
        print("In method", self._testMethodName)

    # Testing the length
    def test_centroid_length(self):
        self.assertTrue(len(clusters_1d), 3)

    # Testing if returned centroids are accurate
    def test_centroid_accuracy(self):
        for i in clusters_1d:
            self.assertTrue((1 <= i <= 9) or (51 <= i <= 69) or (101 <= i <= 111))


if __name__ == '__main__':
    unittest.main()
