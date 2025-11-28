import numpy as np


##### REMEMBER TO NORMALIZE DATASET BEFORE USING THIS CLASS FIT AND PREDICT METHODS
class MyMiniBatchKMeans:
    def __init__(self, n_clusters=8, batch_size=256, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_state)

        self.cluster_centers_ = None

    def _init_centers(self, X):
        """Randomly choose k samples as initial centers."""
        idx = self.random_state.choice(len(X), self.n_clusters, replace=False)
        centers = X[idx].astype(np.float64)
        # normalize for spherical kmeans
        centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
        return centers

    def _closest_center(self, X, centers):
        """
        Based on relationship: ||u-v||^2 = 2 - 2*cos(u,v)  if ||u||=||v||=1 
        Compute nearest cluster with spherical distance:
        dist = sqrt(2 - 2 cos θ)
        Equivalent to maximizing dot(X, centers).
        """
        sims = np.dot(X, centers.T)   # cosine similarity (since normalized)
        return np.argmax(sims, axis=1)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)

        if self.cluster_centers_ is None:
            self.cluster_centers_ = self._init_centers(X)

        counts = np.zeros(self.n_clusters, dtype=np.int64)

        for _ in range(self.max_iter):
            # sample mini-batch
            idx = self.random_state.choice(len(X), self.batch_size, replace=False)
            batch = X[idx]

            # assign points to closest center
            labels = self._closest_center(batch, self.cluster_centers_)

            # update centers using online k-means update rule
            for i, x in zip(labels, batch):
                counts[i] += 1
                eta = 1.0 / counts[i]    # learning rate
                self.cluster_centers_[i] = (1 - eta) * self.cluster_centers_[i] + eta * x
                # re-normalize for spherical kmeans
                self.cluster_centers_[i] /= (
                    np.linalg.norm(self.cluster_centers_[i]) + 1e-12
                )

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._closest_center(X, self.cluster_centers_)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

