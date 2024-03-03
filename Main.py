import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin_min

# Generate random data
X, y = make_blobs(n_samples=300, centers=3, random_state=42)

# Handwritten K-Means clustering
def kmeans_clustering(X, n_clusters, n_iterations=100):
    # Initialize centroids randomly
    initial_centroids = X[np.random.choice(len(X), n_clusters, replace=False)]
    centroids = initial_centroids.copy()

    for _ in range(n_iterations):
        # Assign points to nearest centroid
        labels = pairwise_distances_argmin_min(X, centroids)[0]

        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, initial_centroids, centroids

# Elbow method to find the optimal number of clusters
def elbow_method(X, max_clusters=10):
    distortions = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    return distortions

# Handwritten K-Means clustering for 3 clusters
handwritten_labels, initial_centroids, final_centroids = kmeans_clustering(X, n_clusters=3)

# Plot the handwritten K-Means clustering results with initial and final centroids
plt.scatter(X[:, 0], X[:, 1], c=handwritten_labels, cmap='viridis', edgecolor='k', s=50, label='Final Clusters')
plt.scatter(initial_centroids[:, 0], initial_centroids[:, 1], c='blue', marker='o', s=200, label='Initial Centroids')
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], c='red', marker='X', s=200, label='Final Centroids')
plt.title('Handwritten K-Means Clustering with Initial and Final Centroids')
plt.legend()
plt.show()

# Elbow method plot
distortions = elbow_method(X, max_clusters=10)
plt.plot(range(1, 11), distortions, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.show()

# Using K-Means clustering in sklearn toolkit
kmeans_sklearn = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
sklearn_labels = kmeans_sklearn.labels_

# Compare handwritten K-Means with sklearn toolkit
plt.scatter(X[:, 0], X[:, 1], c=handwritten_labels, cmap='viridis', edgecolor='k', s=50, label='Handwritten K-Means')
plt.scatter(X[:, 0], X[:, 1], c=sklearn_labels, cmap='viridis', marker='x', s=50, label='Sklearn K-Means')
plt.title('Comparison: Handwritten vs. Sklearn K-Means')
plt.legend()
plt.show()

#Image compression with K-Means clustering
image = plt.imread("C:/Users/Alex Joshua Chirwa/Downloads/dog-gbfe9c6841_1920_2.jpg")

plt.figure(figsize=(5, 5))
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')
plt.show()

# Normalize the image to the [0, 1] range
image_normalized = image / 255.0

image_flat = image_normalized.reshape((-1, 3))

n_colors = 16
image_flat_sample = shuffle(image_flat, random_state=42)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=42)
kmeans.fit(image_flat_sample)

labels = kmeans.predict(image_flat)

# Create a compressed image using the cluster centers as colors
image_compressed = kmeans.cluster_centers_[labels].reshape(image.shape)

# Clip the values to the valid range [0, 1]
image_compressed_clipped = np.clip(image_compressed, 0, 1)

plt.figure(figsize=(5, 5))
plt.title('Compressed Image ({} colors)'.format(n_colors))
plt.imshow(image_compressed_clipped)
plt.axis('off')
plt.show()