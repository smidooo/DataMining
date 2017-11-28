import numpy as np
import random
import time


# some constants
K         = 200 # number of clusters
TOLERANCE = 50  # let's accept some differences as convergence


def timestamp(msg):
    t = time.time()
    print(str(round((t - start) / 60.,2)) + ': ' + msg)

def convergence(curr, prev, tolerance):
    
    last_set = set([tuple(x) for x in prev])
    new_set  = set([tuple(x) for x in curr])
    
    diff = len(last_set - new_set)
    timestamp('convergence difference ' + str(diff) + ' of ' + str(len(last_set)))
    
    return (diff <= tolerance)
 
def calculate_clusters(values, centers):
    
    clusters  = {}
    
    for v in values:
        best_cluster = min([(i[0], np.linalg.norm(v-centers[i[0]])) for i in enumerate(centers)],
                            key=lambda t:t[1])[0]
        
        if (best_cluster in clusters):
            clusters[best_cluster].append(v)
        else:
            clusters[best_cluster] = [v]

    # NOTE: this implementation tends to decrease the number of clusters
    #       if they were chosen badly (by random)
    #       very bad because we need a constant number of clusters
    idx = K
    for i in range(len(clusters), K):
        clusters[idx] = random.sample(values, 1)
        idx += 1

    return clusters

def calculate_centers(centers, clusters):
    new_centers = []
    keys = sorted(clusters.keys())
    for k in keys:
        new_centers.append(np.mean(clusters[k], axis = 0))
    return new_centers
    

start = time.time()

def mapper(key, value):
    # key: None
    # value: one line of input file    

    yield "key", value  # this is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.

    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.

    # Initialize to K random centers
    prev_centers = random.sample(values, K)
    curr_centers = random.sample(values, K)

    # iteratively calculate clusters and new centers until convergence
    i = 0
    while not convergence(curr_centers, prev_centers, TOLERANCE):

        prev_centers = curr_centers
        clusters = calculate_clusters(values, curr_centers)
        curr_centers = calculate_centers(curr_centers, clusters)
        
        i += 1
        if (0 == i % 1):
            timestamp('iteration ' + str(i))
        
    output = curr_centers

    timestamp('mapper completed after ' + str(i) + ' iterations')


    yield output
