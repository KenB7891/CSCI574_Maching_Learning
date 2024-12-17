import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def task_1_1_load_data():
    """Task 1.1 load the data for task 1 from the file in the
    data subdirectory named `assg-06-data-kmeans.csv`.  This
    function should return a regular NumPy array of the loaded data.
    We are performing  unsupervised learning tasks in this assignment,
    so this data does not contain any target labels.  So we
    simply return the whole loaded array X as the
    result from this function.

    Params
    ------

    Returns
    -------
    X - A (300,2) shaped NumPy array of data to be used to implement
      and visualize K-means clustering with.
    """
    # your code to load the task 1 data and perform any operations needed goes here
    data = pd.read_csv('../data/assg-06-data-kmeans.csv',header=None)
    X = np.array(data)
    return X


def find_closest_centroids(X, centroids):
    '''This function computes the centroid memberships for every example input
    data in X.  It returns the closest centroids c^(i) for the dataset X, where
    each value in c is the index of the centroid that each input data item was assigned to.

    Params
    ------
    X - A (m, n) shaped Numpy Array.  The input data that we are
        clustering and need to deetermine current centroid membership of.
    centroids - A (K,n) shaped array of centroids positions.  The number of
        features/dimensions n need to match in X and centroids here.

    Returns
    -------
    c - Returns an vector of shape (m,) of the closest centroid positions 0 - K-1
        of each of the m input samples given.
    '''
    indexes = []
    for i in range(len(X[:,0])):
        min_distance = float('inf')
        index = 0
        for j in range(len(centroids[:,0])):
            distance = 0
            for k in range(len(X[0,:])):
                distance += (X[i][k] - centroids[j][k]) ** 2
            if distance < min_distance:
                min_distance = distance
                index = j
        indexes.append(index)
        
    closest_centroid_positions = np.array(indexes)        
    
    return closest_centroid_positions


def compute_centroids(X, c, K):
    '''Return the new centroids by computing the means of the data points
    assigned to each centroid `c`.  We are given X the input data points, and
    the vector c, which is a vector of indexes indicating which centroid each
    data point in X has been assigned too.  The variable K is a regular scalar
    value, indicating the total number of centroids K we are computing.  This
    function should return a new (K, m) shaped array of computed centroids.

    Params
    ------
    X - A (m, n) shaped NumPy array.  The input data that we are
        clustering and need to determine current centroid membership of.
    c - A (m,) shape vector, indicates which centroid  0 - K-1 that each of the
        corresponding sample n was assigned to in an update.
    K - The number of clusters/centroids K we are computing

    Returns
    -------
    centroids - Returns a new array shaped (K, n) of the new updated centroid
       centers for the K clusters of the given data.
    '''
    # your code for this function goes here

    m, n = X.shape
    centroids = np.zeros((K, n))
    
    for i in range(K):
        points = X[c == i]
        centroids[i] = np.mean(points, axis=0)

    return centroids


def kmeans_cluster(X, K, num_iter=10, random_state=42):
    '''Implement our own hand built version of KMeans Clustering.
    As described in the assignment, most of the work has been accomplished
    in the `find_closest_centroids()` and `compute_centroids()` functions.
    This method reuses those function to run a fixed number
    of iterations (specified by num_iter parameter) of finding closest
    centroids and updating to new centroids.  You should create
    a random set of initial centroids by shuffling the X data
    and choosing the K top items from the shuffle to be the
    initial centroids.

    Params
    ------
    X - A (m, n) shaped dataframe/array.  The input data that we are
        clustering and need to determine current centroid membership of.
    K - The number of clusters/centroids K we are to compute and return
    num_iter - The number of fixed iterations of find and update centroids
        to perform, defaults to 10 iterations

    Returns
    -------
    labels - A (m,) shaped NumPy vector that contains the final assigned
       cluster label / indexes from the clustering.  
    history - Returns the history of the centroid selections and updates.
       This should be a regular python list containing Numpy Arrays.  If 
       we iterate 10 times, the returned list will be size 11, because we
       put the initially randomly selected centroids at index 0 of the list.
    '''
    # your implementation of hand-build k-means clustering goes here
    np.random.seed(random_state)
    history = []
    m, n = X.shape
    indices = np.random.choice(m, size=K, replace=False)
    centroids = X[indices]
    history.append(centroids)

    for i in range(num_iter):
        closest_centroids = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, closest_centroids, K)
        history.append(centroids)

    return closest_centroids, history


def task_2_1_load_data():
    """Task 2.1 load the data for task 2 from the file in the
    data subdirectory named `assg-06-data-pca.csv`.  This
    function should return a regular NumPy array of the loaded data.
    We are performing  unsupervised learning tasks in this assignment,
    so this data does not contain any target labels.  So we
    simply return the whole loaded array X as the
    result from this function.

    Params
    ------

    Returns
    -------
    X - A (50,2) shaped NumPy array of data to be used to implement
      and visualize principal compoment analysis with.
    """
    # your code to load the task 2 data and perform any operations needed goes here
    data = pd.read_csv('../data/assg-06-data-pca.csv',header=None)
    X = np.array(data)
    return X


def feature_normalize(X):
    '''Given a matrix of size m x n (m sample data points, with n features for
    each data point), perform both mean normalization and feature scaling on
    all of the features for all of the m sample data points. This function returns
    the normalized X data, and it also returns the original mean and variances
    found for each feature.

    Params
    ------
    X - A set of unbaleled data of some shape (m, n) to be normalized and
        scaled.

    Returns
    -------
    X_norm - The normalized and scaled data.  Each feature m will have a mean
       of 0 and a standard deviation of 1 after transformation.
    mu - A numpy vector of shape (n),) that contains the original means of each
       of the m features in the input data
    sigma - A numpy vector of shape (n,) that contains the original variances
       (standard deviations) of each of the m features in the input data
    '''
    # implementation of normalization and scaling goes here, make sure
    # you perform by hand, do not use scikit-learn normalization objects
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    X_normalized = X - mu
    X_normalized_scaled = (X - mu) / sigma
    
    return X_normalized_scaled, mu, sigma


def pca(X):
    '''This function performs the principal component analysis (PCA) algorithm
    on the data.  It expects raw (unnormalized and unscaled) data as input.
    It will first normalize and scale the data.  It then computes the
    covariance matrix and uses singular value decomposition to compute
    the U,S,V matrices of the decomposition. It returns the calculated principal
    component vectors U, as well as the array of variances S explained by each
    principal component.  In addition the mu and sigma means and variances of
    the original unnormalized data are returned if needed to project back onto
    the original data space.

    Params
    ------
    X - A set of unbaleled data of some shape (m, n) to be normalized and
        scaled.

    Returns
    -------
    U - The resulting principal components vectors matrix U from singular value
        decomposition of the covariance matrix of the original data
    S - The resulting explained variances vector S of the singular value 
        decomposition of the input data.
    mu - A vector of shape (n,) giving the original means of each of the n features
         of the given input data X
    sigma - A vector of shape (n,) giving the original variances of each of the
         n features of the given input data X
    '''
    # implementation of hand-built PCA goes here
    X_norm_scaled, mu, sigma = feature_normalize(X)

    m, n = X_norm_scaled.shape

    covariance_matrix = (1.0 / m) * np.dot(X_norm_scaled.T, X_norm_scaled)
    U, S, V = np.linalg.svd(covariance_matrix)
    
    return U, S, mu, sigma


def project_data(X, U, K):
    '''Computes the reduced data representation by projecting X onto the
    top K principal components defined by the eigenvectors in U.  If X is
    an m x n matrix of input data, this will return the projected matrix Z
    which will be reduced to m x K, where K < n the number of original features.

    Params
    ------
    X - Data of shape (m,n) to be projected into the first K principle componenents
        defined by U 
    U - Principal components of dataspace X as determined by a PCA/SVD
    K - The number of components to project the X data down to, where K needs to
        be less than n, the number of feature dimensions of the data, for this
        operation to be meaningful.

    Returns
    -------
    Z - The projected samples of X into K dimensions adccording to the principal
        component vectors of U.  The shape of Z should be (m, K) where K is less
        than the original n numebr of dimensions of the input data X, and each
        sample in X now has just K columns/dimensions instead of the original n.
    '''
    # implementation of PCA dimensionality reduction / projection goes here

    # first reduce U (the matrix of principle component vectors) to k dimensions (by taking thr first k column vectors)
    U_reduce = U[:,:K]

    # perform projection
    Z = np.dot(X, U_reduce)
    
    return Z


def recover_data(Z, U):
    '''Recovers an approximation of the original data when using the projected
    data.  We are given the projection Z and the principal component vectors U.
    We return the approximate recovered/projected position in the normalized
    space for the data.  The return recovered values X_rec will have the
    same dimensions as the X_norm normalized values, but points in the sample
    will be the positions after projecting down to 1 dimension and then
    recovering back to 2 dimensions.

    Params
    ------
    Z - The samples projected down into K dimensions, the result of performing
        a project_data() function.  The array is expected to be of shape (m,K)
        where each original sample is a row that was projected down to the K
        principal components.
    U - Principal component eigenvectors of dataspace X as determined by a PCA/SVD
        The size of the original space n can be determined from this matrix as
        it should be an (n,n) shaped square matrix.

    Returns
    -------
    X_rec - Returns original samples reprojected back into n dimensions as 
       best as could be determined from the reduced positon Z and the principal 
       components U
    '''
    # implementation of PCA dimensionality data recovery  goes here
    m, K = Z.shape

    # reduce U again
    U_reduce = U[:,:K]

    
    X_rec = np.dot(Z, U_reduce.T)
    
    return X_rec
