import numpy as np
import random
from scipy.spatial.distance import squareform, pdist, cdist
from skimage.util import img_as_float

### Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centroids = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        ### YOUR CODE HERE
        # 1. Assign each point to the closest centroid
        assignments = assign(features,centroids,k)
        old_Centroids = centroids.copy()
        # 2. Compute new centroid of each cluster
        centroids = get_New_Centroids(features,assignments,k)
        # 3. Stop if cluster assignments did not change
        if is_Converged(old_Centroids,centroids,k):
            break
        ### END YOUR CODE

    return assignments

# this method assigns labels to each feature, it loops over the feaures, and calculates all distances and choose the closest
# this methods uses two nested loops for the assign task
def assign(features,centroids,k) :
    N, _ = features.shape
    assignments = np.zeros(N)
    for i in range(N):
        distances = np.zeros(k)
        for j in range(len(centroids)):
            distances[j] = euclidean_distance(features[i],centroids[j])
        closest_distance = np.argmin(distances)   
        assignments[i] = closest_distance
    return assignments 

# this method gets the new centroids for all features
def get_New_Centroids(features,assignments,k) :
    centroids = [[] for _ in range(k)]
    for i in range(k):
        centroids[i] = np.mean(features[assignments == i], axis=0)
    return centroids    

# this method checks if the previous centroids are the same new ones, by calculating the distance between each old and new centroid
# and if sum of all distances are zero, then we need to stop.
def is_Converged(old_Centroids, new_Centroids,k):
    distances = [[] for _ in range(k)]
    for i in range(k) :
        distances[i] = euclidean_distance(old_Centroids[i],new_Centroids[i])
    return sum(distances) == 0    

# this calculates the euclidean distance between two vectors represent the feature and the centroid.
def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1 - x2)**2, axis=0))

# this method makes use of cdist function to calculate distances between each feature and all centroids fast
def assign_Fast(features,centroids,k):
    N, _ = features.shape
    assignments = np.zeros(N)
    distances = cdist(features,centroids)
    assignments = np.argmin(distances,axis=1)
    return assignments    

# this method normalizes the features
def normalize_Features(features) :
    mean = np.mean(features,axis = 0)
    standard_deviation = np.std(features,axis = 0)
    return (features - mean)/standard_deviation

def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find cdist (imported from scipy.spatial.distance) and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centroids = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        ### YOUR CODE HERE
        # 1. Assign each point to the closest centroid using assign_Fast method
        assignments = assign_Fast(features,centroids,k)
        old_Centroids = centroids.copy()
        # 2. Compute new centroid of each cluster
        centroids = get_New_Centroids(features,assignments,k)
        # 3. Stop if cluster assignments did not change
        if is_Converged(old_Centroids,centroids,k):
            break
        ### END YOUR CODE

    return assignments



def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to define distance between clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Hints
    - You may find pdist (imported from scipy.spatial.distance) useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """



    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N, dtype=np.uint32)
    centers = np.copy(features)
    n_clusters = N
    

    while n_clusters > k:
        ### YOUR CODE HERE
        pass
        ### END YOUR CODE

    return assignments


### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))

    ### YOUR CODE HERE
    # All we need is to flatten/reshape the image array, so that each pixel will be treated as 
    # an independent array, and this can be done using "reshape" method
    features = img.reshape(H*W, C)
    ### END YOUR CODE

    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).

    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    ### YOUR CODE HERE
    # First, we need to store all possible combinations between x,y to represent all positions, so that we can add them to our 
    # feature vector. so, the array of all possible locations will look like 
            #                            [[0,0],[0,1],[0,2],......,[0,n],
            #                             [1,0],[1,1],[1,2],......,[1,n],
            #                                         .
            #                                         .
            #                             [n,0],[n,1],[n,2],......,[n,n]]

    # this can be done using "np.dstack" method which combines an array and stores it like a stack
    # however, to be able to use np.dstack method and reach to our goal array, we need the input to be like
            #                            [[0,0,0,...,1,1,1,.......,n,n,n],
            #                             [0,1,2,...,0,1,2,.......,0,1,2]] 
        
    # And the above array can be formed using "np.mgrid" method which returns mesh-grid ndarrays all of the same dimensions,  
    # and then reshaping it to be (2,h*w), so the steps are as follows :-
    # 1-create mesh-grid ndarrays
    # 2-reshape it to be (2,h*w)
    # 3-use np.dstack to have all locations
    mesh_grid = np.mgrid[0 : H, 0 : W]
    # the output will be 
                                    #array([[[  0,   0,   0, ...,   0,   0,   0],
                                    #        [  1,   1,   1, ...,   1,   1,   1],
                                    #        [  2,   2,   2, ...,   2,   2,   2],
                                    #        ...,
                                    #        [396, 396, 396, ..., 396, 396, 396],
                                    #        [397, 397, 397, ..., 397, 397, 397],
                                    #        [398, 398, 398, ..., 398, 398, 398]],
                                    #
                                    #       [[  0,   1,   2, ..., 621, 622, 623],
                                    #        [  0,   1,   2, ..., 621, 622, 623],
                                    #        [  0,   1,   2, ..., 621, 622, 623],
                                    #        ...,
                                    #        [  0,   1,   2, ..., 621, 622, 623],
                                    #        [  0,   1,   2, ..., 621, 622, 623],
                                    #        [  0,   1,   2, ..., 621, 622, 623]]])
    mesh_grid_reshaped = mesh_grid.reshape(2,H*W)                                    
    # the output will be 
                                    #array([[  0,   0,   0, ..., 398, 398, 398],
                                    #       [  0,   1,   2, ..., 621, 622, 623]])
    xy_combinations = np.dstack(mesh_grid_reshaped)        
    # the final output, which contains all possible combinations of locations, will be
                                    #array([[[  0,   0],
                                    #        [  0,   1],
                                    #        [  0,   2],
                                    #        ...,
                                    #        [398, 621],
                                    #        [398, 622],
                                    #        [398, 623]]])
                            
    # now, we have the positions to be stored,so let's prepare the colors to be stored also
    # color shape is (h,w,3), so let's flatten/reshape this array to be (h*w,3), in this way, we have each pixel color
    # as an independent array (r,g,b), and this will make it easy for us to concatenate them to feature vector later
    color_reshaped = color.reshape((H * W, 3))
    # now, let's concatenate color and position for each pixel to the final feature vector
    features[:,0:C] = color_reshaped
    features[:,C:C+2] = xy_combinations
    # normalize features
    features = normalize_Features(features)
    ### END YOUR CODE

    return features

### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    ### YOUR CODE HERE
    #---------------------------------------------------------------------------------------
    # First approach using nested loop to search for pixels where mask and mask_gt agree
    count = 0
    for i in range(len(mask)) : 
        for j in range(len(mask[0])) :
            if(mask[i][j] == mask_gt[i][j]):
                count += 1
    
    #accuracy = (count/no_of_pixels)   
    #---------------------------------------------------------------------------------------
    #Second approach makes use of numpy mean function
    accuracy = np.mean(mask_gt == mask)        
    #---------------------------------------------------------------------------------------
    ### END YOUR CODE

    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments.
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
