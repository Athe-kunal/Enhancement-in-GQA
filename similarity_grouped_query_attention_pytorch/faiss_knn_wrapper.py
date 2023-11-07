# import numpy as np
from sklearn.utils.validation import check_is_fitted


def is_available():
    try:
        import faiss
        return True
    except ImportError:
        return False


class FaissKNNClassifier:
    """ Scikit-learn wrapper interface for Faiss KNN.

    Parameters
    ----------
    n_neighbors : int (Default = 5)
                Number of neighbors used in the nearest neighbor search.

    n_jobs : int (Default = None)
             The number of jobs to run in parallel for both fit and predict.
              If -1, then the number of jobs is set to the number of cores.

    algorithm : {'brute', 'voronoi'} (Default = 'brute')

        Algorithm used to compute the nearest neighbors:

            - 'brute' will use the :class: `IndexFlatL2` class from faiss.
            - 'voronoi' will use :class:`IndexIVFFlat` class from faiss.
            - 'hierarchical' will use :class:`IndexHNSWFlat` class from faiss.

        Note that selecting 'voronoi' the system takes more time during
        training, however it can significantly improve the search time
        on inference. 'hierarchical' produce very fast and accurate indexes,
        however it has a higher memory requirement. It's recommended when
        you have a lots of RAM or the dataset is small.

        For more information see: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

    n_cells : int (Default = 100)
        Number of voronoi cells. Only used when algorithm=='voronoi'.

    n_probes : int (Default = 1)
        Number of cells that are visited to perform the search. Note that the
        search time roughly increases linearly with the number of probes.
        Only used when algorithm=='voronoi'.

    References
    ----------
    Johnson Jeff, Matthijs Douze, and Hervé Jégou. "Billion-scale similarity
    search with gpus." arXiv preprint arXiv:1702.08734 (2017).
    """

    def __init__(self,
                 n_neighbors=5,
                 n_jobs=None,
                 algorithm='brute',
                 n_cells=100,
                 n_probes=1,
                 device="cpu"):

        self.n_neighbors = n_neighbors
        assert self.n_neighbors>0, "Number of neighbors should be greater than 0" 
        self.n_jobs = n_jobs
        self.algorithm = algorithm
        assert self.algorithm in ['brute', 'voronoi', 'hierarchical'], f"Invalid algorithm option. Expected ['brute', 'voronoi', 'hierarchical'], got {self.algorithm}" 
        self.n_cells = n_cells
        self.n_probes = n_probes
        import faiss
        self.faiss = faiss
        if device == "cpu":
            self.cuda = False
            self.device = None
        else:
            self.cuda = True
            if ":" in device:
                self.device = int(device.split(":")[-1])
            else:
                self.device = 0

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """Finds the K-neighbors of a point.

        Parameters
        ----------

        X : array of shape (n_samples, n_features)
            The input data.

        n_neighbors : int
            Number of neighbors to get (default is the value passed to the
            constructor).

        return_distance : boolean, optional. Defaults to True.
            If False, distances will not be returned

        Returns
        -------
        dists : list of shape = [n_samples, k]
            The distances between the query and each sample in the region of
            competence. The vector is ordered in an ascending fashion.

        idx : list of shape = [n_samples, k]
            Indices of the instances belonging to the region of competence of
            the given query sample.
        """
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        check_is_fitted(self, 'index_')
        assert X.shape>=2, f"DimensionError: The X values should have atleast a dimension of 2"
        dist, idx = self.index_.search(X, n_neighbors)
        if return_distance:
            return dist, idx
        else:
            return idx

    def fit(self, X):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Data used to fit the model.

        y : array of shape (n_samples)
            class labels of each example in X.
        """
        # X = np.atleast_2d(X).astype(np.float32)
        assert len(X.shape)>=2, f"DimensionError: The X values should have atleast a dimension of 2"
        X = X.contiguous()
        d = X.shape[1]  # dimensionality of the feature vector
        self._prepare_knn_algorithm(X, d)
        self.index_.add(X)
        return self
    def _prepare_knn_algorithm(self,X, d: int) -> None:
        """Create the faiss index.

        Args:
            d: feature dimension
        """
        if self.cuda:
            self.res = self.faiss.StandardGpuResources()
            self.config = self.faiss.GpuIndexFlatConfig()
            self.config.device = self.device

            if self.algorithm == 'brute':
                self.index_ = self.faiss.GpuIndexFlatL2(self.res, d, self.config)
            elif self.algorithm == 'voronoi':
                quantizer = self.faiss.GpuIndexFlatL2(self.res,d,self.config)
                self.index_ = self.faiss.GpuIndexIVFFlat(quantizer, d, self.n_cells)
                self.index_.train(X)
                self.index_.nprobe = self.n_probes
            elif self.algorithm == 'hierarchical':
                self.index_ = self.faiss.IndexHNSWFlat(d, 32)
                self.index_.hnsw.efConstruction = 40
        else:
            self.index_ = self.faiss.IndexFlatL2(d)
            if self.algorithm == 'brute':
                self.index_ = self.faiss.IndexFlatL2(d)
            elif self.algorithm == 'voronoi':
                quantizer = self.faiss.IndexFlatL2(d)
                self.index_ = self.faiss.IndexIVFFlat(quantizer, d, self.n_cells)
                self.index_.train(X)
                self.index_.nprobe = self.n_probes
            elif self.algorithm == 'hierarchical':
                self.index_ = self.faiss.IndexHNSWFlat(d, 32)
                self.index_.hnsw.efConstruction = 40
        