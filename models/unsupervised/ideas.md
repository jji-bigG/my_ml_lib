# Unsupervised Learning Techniques

### 1. Clustering

#### K-Means Clustering

- **Description**: Partitions the data into \( k \) clusters, where each data point belongs to the cluster with the nearest mean.
- **Use Cases**: Market segmentation, image compression, document clustering.

#### Hierarchical Clustering

- **Description**: Builds a hierarchy of clusters either through a bottom-up (agglomerative) or top-down (divisive) approach.
- **Use Cases**: Gene sequence analysis, customer segmentation.

#### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

- **Description**: Clusters data based on the density of data points, identifying clusters of varying shapes and sizes and marking outliers as noise.
- **Use Cases**: Geographic data analysis, noise detection in data.

### 2. Dimensionality Reduction

#### Principal Component Analysis (PCA)

- **Description**: Reduces the dimensionality of data by transforming it to a new set of variables (principal components) that are orthogonal and capture the maximum variance.
- **Use Cases**: Data visualization, noise reduction, feature extraction.

#### t-Distributed Stochastic Neighbor Embedding (t-SNE)

- **Description**: Reduces high-dimensional data to two or three dimensions for visualization by modeling the similarity between pairs of data points.
- **Use Cases**: Visualizing high-dimensional datasets, image recognition.

#### Independent Component Analysis (ICA)

- **Description**: Separates a multivariate signal into additive, independent non-Gaussian components.
- **Use Cases**: Signal processing, source separation (e.g., separating audio signals).

### 3. Association Rule Learning

#### Apriori Algorithm

- **Description**: Identifies frequent itemsets in transactional data and generates association rules.
- **Use Cases**: Market basket analysis, recommendation systems.

#### Eclat Algorithm

- **Description**: Similar to the Apriori algorithm but uses a depth-first search to find frequent itemsets.
- **Use Cases**: Market basket analysis, recommendation systems.

### 4. Anomaly Detection

#### Isolation Forest

- **Description**: Isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.
- **Use Cases**: Fraud detection, network security.

#### One-Class SVM

- **Description**: Identifies outliers by learning a decision function for anomaly detection.
- **Use Cases**: Novelty detection, fraud detection.

### 5. Self-Organizing Maps (SOM)

- **Description**: A type of artificial neural network that uses unsupervised learning to produce a low-dimensional (typically 2D) representation of the input space, preserving the topological properties.
- **Use Cases**: Data visualization, feature mapping.

### 6. Gaussian Mixture Models (GMM)

- **Description**: Models the data as a mixture of multiple Gaussian distributions, where each component represents a cluster.
- **Use Cases**: Clustering, density estimation, pattern recognition.

### 7. Autoencoders

- **Description**: A type of neural network used to learn efficient codings of input data, typically for dimensionality reduction or feature learning.
- **Use Cases**: Image denoising, anomaly detection, data compression.

### 8. Non-Negative Matrix Factorization (NMF)

- **Description**: Factorizes a non-negative matrix into the product of two non-negative matrices, often used for parts-based representation of data.
- **Use Cases**: Topic modeling, recommendation systems, image processing.

### 9. Spectral Clustering

- **Description**: Uses the eigenvalues of a similarity matrix to reduce dimensionality before clustering in fewer dimensions.
- **Use Cases**: Image segmentation, community detection in graphs.

### 10. BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)

- **Description**: An efficient clustering method for very large datasets, which incrementally and dynamically clusters incoming data points.
- **Use Cases**: Large-scale data clustering, real-time data clustering.

These techniques are commonly used in various fields such as data mining, computer vision, natural language processing, bioinformatics, and more to uncover hidden structures in data without requiring labeled examples.
