import numpy as np
import random
import time


# some constants
K            = 200 # number of clusters
INIT_RANDOM  = 100   # select some points randomly (to speed up)
TOLERANCE    = 80  # let's accept some differences as convergence


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
        
        best_l2 = np.linalg.norm(v - centers[0])
        best_cluster = 0
        for i in enumerate(centers):
            l2 = np.linalg.norm(v-centers[i[0]])
            if (l2 < best_l2):
                best_l2 = l2
                best_cluster = i[0]
                
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

def lloyds(cur, prev, val):
    i = 0
    while not convergence(cur, prev, TOLERANCE):

        prev= cur
        clusters = calculate_clusters(val, cur)
        cur= calculate_centers(cur, clusters)
        
        i += 1
        if (0 == i % 1):
            timestamp('iteration ' + str(i))

    return cur

def calc_norm_to_b(b, values):
    # calculate all norms of the values to a center b
    l2 = []
    for v in values:
        l2.append(np.linalg.norm(v-b))
    return l2


def update_l2_min(L2, b, values):
    l2 = calc_norm_to_b(b, values)
    L2 = np.minimum(L2, l2)
    return L2


def initialize_d2(values):
    # set the first N_RANDOM centers randomly and calculate the L2 matrix
    B = random.sample(values, INIT_RANDOM)
    L2 = calc_norm_to_b(B[0], values)
    for b in B[1:]:
        L2 = update_l2_min(L2, b, values)
    return B, L2


def d2_sampling(values):
    B, l2_min = initialize_d2(values)
    
    for i in range(INIT_RANDOM, K):
        max_v_ind = np.argmax(l2_min)
        b = values[max_v_ind]
        print('center ' + str(len(B)) + ' is value number ' + str(max_v_ind) + ' with max norm ' + str(np.max(l2_min)))
        B.append(b)
        l2_min = update_l2_min(l2_min, b, values)
    return np.array(B)

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

    print('REDUCER: shape of input:' + str(np.shape(values)))

    # Initialize to K centers
    prev_centers = random.sample(values, K)
    curr_centers = d2_sampling(values)
    
    print('shape of centers: ', np.shape(curr_centers))

    # iteratively calculate clusters and new centers until convergence
    output = lloyds(curr_centers, prev_centers, values)

    timestamp('reducer completed')


    yield output
