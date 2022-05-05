---
date created: 2022-03-08 10:29
date updated: 2022-05-05 16:19
tags:
  - '#clusters'
  - '#Clustering'
  - '#algorithm'
  - '#Dimensionality-Reduction'
  - '#dimensionality-reduction'
---

# Unsupervised Learning

```ad-definition
Building a model using examples **without labels** that learns to predict new examples
```

Given $T = {x_1, x_m}$ ^[no $y$ because there are no labels] output a hidden structure behind the various $x$ values, that being the #clusters .

## #Clustering

Find a function $f \in \mathbb{N}^X$ that assigns each input $x \in X$ a cluster index $f(x) \in \mathbb{N}$. All the points mapped to the same index form a cluster.
![[unsupervised-learning-clustering.png]]

### Why?

Clustering allows us to analyse data by grouping together data points that exhibit some regular pattern or similarity under some predefined criterion, and it can be used to compress data by reducing the number of data points as opposed to reducing the feature dimensionality.

There are several situations in which it could be applied, such as:

- Clustering users by preference^[e.g., based on movie ratings]
- Grouping gene families from gene sequences
- Finding communities on social networks
- Image segmentation^[distinguishing different parts or layers of an image]
- Anomaly detection
  - Analyse a set of events or objects and flag some of them as being unusual or atypical^[credit card fraud detection, video surveillance]

### K-Means Clustering
Given data points $X =[x_1, ..., x_n] \in \mathbb{R}^{d \times n}$, pix a number of clusters $k$ and find a partition of data points into $k$ sets minimising the variation within each set: $$\min_{C_1, ..., C_k}\sum_{j = 1}^{k}V(C_j)$$
The variation $V(C_j)$ is typically given by $\sum_{i \in C_j}||x_i - \mu_j||^2$ and the centroid $\mu_j$ is computed with $\frac{1}{|C_j|}\sum_{i \in C_j} x_i$.

#### Properties
```ad-note
Note that K-means clustering is sensitive to the scale of the features. Feature normalisation is a necessity in the case of features with different scales.
```
The K-means clustering algorithm is guaranteed to converge, as it strictly improves the objective if there is at least one cluster change and the set of possible partitions is finite. 
```ad-question
title: Does each step of k-means move towards reducing the loss function (or at least not increasing it)?
Idea:
1. Assignment - Any other assignment would lead to a larger loss
2. Centroid computation - The mean of a set of values minimises the squared error.
```

It is not, on the other hand, guaranteed to find the global minimum; only a local one.
```ad-question
title: Does this mean that k-means will always find the minimum loss/clustering?
No. It will find *a* minimum. Unfortunately, the k-means loss function is generally not convex and for most problems has many minima. We are only guaranteed to find one of them.
```
#### Initial Centroid Selection
Results can vary drastically based on random seed selection; some seeds can result in a poor convergence rate or in convergence to suboptimal clusterings. Some common heuristics include:
- Random points^[not examples] in the space
- Randomly pick examples
- Points least similar to any existing centre^[furthest centres heuristic]
- Try out multiple starting points
- Initialise with the results of another clustering method

#### Running Time
For step 1, that is the assignment step, the time complexity is $O(kn)$. For step 2, the centroid computation, the time complexity is $O(n)$.

## #Dimensionality-Reduction

Find a function $f \in Y^X$ mapping each (high-dimensional) input $x \in X$ to a lower dimensional embedding $f(x) \in Y$, where $dim(Y) \ll dim(X)$.

### Why?

Using #dimensionality-reduction we can compress the data by reducing the feature dimensionality while still preserving as much data as possible^[the way information loss is measured yields different algorithms]. This reduces the time needed for subsequent data elaboration and/or storage and allows for better visualisation of the data as well as reducing the curse of dimensionality.

## Density Estimation

Find a probability distribution $f \in \Delta(X)$ that fits the data $x \in X$.

### Why?

Density estimation is a method used to get an explicit estimate of the unknown probability distribution that generated the training data, thus enabling not only the generation of new data by sampling from the estimated distribution but also the detection of anomalies/novelties in terms of data points that exhibit low probabilities according to the estimated distribution.

## Principal Component Analysis^[PCA]

### Calculating the i-th Principal Component

Given $w_i \in argmax\{w^T Cw : w^T w = 1, w \perp w_j for\ 1 \le j \lt i\}$, where $w^T Cw$ shows the variance along $w$ in the below graph, it can be shown that i-th largest eigenvalue of $C$ is the variance along the i-th principal component, which itself is the corresponding eigenvector.

![[unsupervised-learning-first-principal-component.png]]

```ad-note
title: Alternative Interpretation
The first principal component can be interpreted as the line in space with minimal squared distance from the data points.
```

### Dimensionality Reduction Using PCA

Let $\hat W = [w_1, ..., w_k]$ hold the first $k$ principal components derived from data points $\bar X = [\bar x_1, ..., \bar x_n]$. We change this to a reduced coordinate system with the $k$ principal components as axes. We have $T = \hat W^T \bar X \in \mathbb{R}^{k \times n}$ where $T$ is the set of principal component scores.

### How Many Principal Components?
The number of components for dimensionality reduction depends on the goal and application. There are no ways of validating it unless we use it in the context of a supervised method^[e.g., reducing the input dimensionality for a supervised algorithm]. Despite this we can compute the cumulative proportion of explained variance, which for the first principal components is given by $\frac{\sum_{j = 1}^{k}\lambda_j}{\sum_{j = 1}^{m}C_{jj}}$ in the case of eigenvalue decomposition, and by $\frac{\sum_{j = 1}^{k}s_{j}^{2}}{\sum_{ij}\bar X_{ji}^{2}}$ in the case of SVD^[Singular Value Decomposition]. This all allows us to estimate the amount of information loss.

## Deep Generative Models
