import numpy as np
from collections import defaultdict

def k_means_clustering(
    points: list[tuple[float, ...]], 
    k: int, 
    initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
    # Your code here

    # initial centroids placement
    new_centroids = initial_centroids        

    for _ in range(max_iterations):
        points_centroid = []
        for point in points:
            temp_distance = []
            for centroid in new_centroids:
                temp_distance.append(distance(point, centroid))
            points_centroid.append(temp_distance)

        assigned_centroids = [np.argmin(centroid) for centroid in points_centroid]
        
        # new initial centroids
        dict_centroids_points = defaultdict(list)
        for point, assigned_centroid in zip(points, assigned_centroids):
            dict_centroids_points[assigned_centroid].append(point)
        new_centroids = []

        for i in range(k):
            centroid = np.mean(dict_centroids_points[i], axis=0)
            new_centroids.append(tuple(float(x) for x in centroid))

    return new_centroids

def distance(p1, p2):
    temp = 0
    for c1, c2 in zip(p1, p2):
        temp += (c1-c2)**2
    
    return np.sqrt(temp)