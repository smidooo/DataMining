import numpy as np
import random
import time


def convergence(curr, prev):
    last_set = set([tuple(x) for x in prev])
    new_set  = set([tuple(x) for x in curr])
    return (last_set == new_set)
 
def calculate_clusters(values, centers):
    clusters  = {}
    for v in values:
        best_cluster = min([(i[0], np.linalg.norm(v-centers[i[0]])) for i in enumerate(centers)],
                            key=lambda t:t[1])[0]
        
        if (best_cluster in clusters):
            clusters[best_cluster].append(v)
        else:
            clusters[best_cluster] = [v]
            
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
    prev_centers = random.sample(values, 300)
    curr_centers = random.sample(values, 300)
    i = 0
    while not convergence(curr_centers, prev_centers):

        prev_centers = curr_centers
        clusters = calculate_clusters(values, curr_centers)
        curr_centers = calculate_centers(curr_centers, clusters)
        
        i += 1
        if (0 == i % 10):
            t = time.time()
            print('iteration ' +  str(i) + ' after ' + str(round((t - start) / 60.,2)) + ' minutes.')
        
    output = curr_centers

    end = time.time()
    print('completed after ' + str(i) + ' iterations in ' + str(round((end - start) / 60.,2)) + ' minutes.')

    yield output
