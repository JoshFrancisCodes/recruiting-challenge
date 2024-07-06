from scipy.spatial import procrustes
from scipy.spatial.distance import euclidean, directed_hausdorff
from fastdtw import fastdtw

def hausdorff_distance(shape1, shape2):
    return max(directed_hausdorff(shape1, shape2)[0], directed_hausdorff(shape2, shape1)[0])

def procrustes_analysis(shape1, shape2):
    mtx1, mtx2, disparity = procrustes(shape1, shape2)
    return disparity

def dtw_distance(shape1, shape2):
    distance, path = fastdtw(shape1, shape2, dist=euclidean)
    return distance
