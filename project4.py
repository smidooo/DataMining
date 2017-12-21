import numpy as np

iterations = 0
last_recommendation = None
last_user_features = None
M = {}
b = {}
w = {}
Q = {}

C_ALPHA = 0.2 #100 works

def set_articles(articles):
    # method is called once at the beginning
    # all (80) articles are provided (i.e. the ID and features)
    global M, b, w, Q

    # initialization or article features, M and b
    for a in articles:
        M[a] = np.identity(6).reshape(6,6)
        b[a] = np.zeros(6).reshape(6,1)
        w[a] = np.zeros((6,1))
        Q[a] = M[a]


def update(reward):
    # called after providing an article to the user
    # update the policy if necessary
    # reward:
    #  -1: recommended and displayed article didn't match
    #   0: recommended article has been displayed, but user didn't click
    #   1: recommended article has been displayed, the user has clicked

    global M, b, w, Q, last_recommendation, last_user_features, iterations

    if ((1000 < iterations) and (0 <= reward)):

        M_diff = np.multiply(last_user_features.reshape(6,1), last_user_features.reshape(6,1).T)
        M[last_recommendation] = M[last_recommendation] + (M_diff)
        b[last_recommendation] = b[last_recommendation] + (last_user_features*reward)

        Q[last_recommendation] = np.linalg.inv(M[last_recommendation]).reshape(6,6)
        w[last_recommendation] = np.dot(Q[last_recommendation], b[last_recommendation]).reshape(6,1)

    iterations += 1


def recommend(time, user_features, choices):
    # called for every line in the logs
    # select one article from choices, return it (i.e. recommend for the user)
    # update method will be called with the result (i.e. click / no click)

    global M, b, w, Q, last_recommendation, last_user_features
    
    last_user_features = np.array(user_features).reshape(6,1)
    max_ucb = -1

    #for all x actoins in action set A_t (choices)
    for x in choices:        
        
        proj = np.dot(np.dot(last_user_features.T, Q[x]).reshape(1,6), last_user_features)
        ucb = np.dot(w[x].T, last_user_features) + (C_ALPHA * np.sqrt(proj))

        if (max_ucb < ucb[0][0]):
            max_ucb = ucb[0][0]
            last_recommendation = x

    return last_recommendation
