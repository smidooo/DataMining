import numpy as np
import time

"""
structure of logs (100'000):
----------------------------
 - timestamp
 - user features (6d vector of doubles)
 - available articles (list of IDs)
1241160900
1.000000 0.000012 0.000000 0.000006 0.000023 0.999958
109513 0 109498 109509 109508 109473 109503 109502 109501 109492 109495 109494 109484 109506 109510 109514 109505 109515 109512 109513 109511 109453
structure of articles (80):
---------------------------
 - article ID
 - list of features (6)
109568
0.336694544019 0.403501479737 0.404825199717 0.129586091883 0.0890207306141 0.540844109036
"""

iteration = 0
nof_user_features = 6
nof_articles = None
article_features = None
last_recommendation = None
last_user_features = None
M = {}
b = {}
w = {}
Q = {}
success = {}

C_ALPHA = 0.05 #100 works

start = time.time()

def set_articles(articles):
    # method is called once at the beginning
    # all (80) articles are provided (i.e. the ID and features)
    global nof_user_features, nof_articles, article_features, M, b, w, Q

    # initialization or article features, M and b
    article_features = articles
    #nof_articles = len(article_features)
    for a in articles:
        M[a] = np.identity(nof_user_features).reshape(6,6)
        b[a] = np.zeros(nof_user_features).reshape(6,1)
        w[a] = np.zeros((6,1))
        Q[a] = M[a]


def update(reward):
    # called after providing an article to the user
    # update the policy if necessary
    # reward:
    #  -1: recommended and displayed article didn't match
    #   0: recommended article has been displayed, but user didn't click
    #   1: recommended article has been displayed, the user has clicked

    global M, b, w, Q, last_recommendation, last_user_features, success, start, iteration

    #print('   ==> ' + str(reward))
    
    if (0 <= reward):

        M_diff = np.multiply(last_user_features.reshape(6,1), last_user_features.reshape(6,1).T)
        M[last_recommendation] = M[last_recommendation] + (M_diff)
        b[last_recommendation] = b[last_recommendation] + (last_user_features*reward)

        Q[last_recommendation] = np.linalg.inv(M[last_recommendation]).reshape(6,6)
        w[last_recommendation] = np.dot(Q[last_recommendation], b[last_recommendation]).reshape(6,1)
        
    
    #if (0 <= reward):
    #    success[str(last_user_features)] = last_recommendation

    #iteration += 1
    #if (100000 == iteration):
    #    t = time.time()
    #    print(str(round((t - start) / 60.,2)) + ': completed')

    #print('Mdiff', np.shape(M_diff), M_diff)

    #print('M', last_recommendation, np.shape(M[last_recommendation]), M[last_recommendation])
    #print('b',last_recommendation,np.shape(b[last_recommendation]), b[last_recommendation])
    #print(np.shape(last_user_features), last_user_features)


def recommend(time, user_features, choices):
    # called for every line in the logs
    # select one article from choices, return it (i.e. recommend for the user)
    # update method will be called with the result (i.e. click / no click)

    #print ('recommend')
    global M, b, w, Q, last_recommendation, last_user_features, success
    
    last_user_features = np.array(user_features).reshape(6,1)
    max_ucb = -1

    #luf = str(last_user_features)
    #if ((luf in success.keys()) and (success[luf] in choices)):
    #    print('reuse: ' + str(success[luf])),
    #    return success[luf]

    #print('M',np.shape(M))
    #print('b',np.shape(b))
    #print('last_user_features', np.shape(last_user_features), last_user_features)


    #for all x actoins in action set A_t (choices)
    for x in choices:        
        
        proj1 = np.dot(last_user_features.T, Q[x]).reshape(1,6)
        #print('pro1', np.shape(proj1))
        proj = np.dot(proj1, last_user_features)
        #print('proj', np.shape(proj))

        ucb = np.dot(w[x].T, last_user_features) + (C_ALPHA * np.sqrt(proj))
        #print('w', np.shape(w[x]),w[x])
        #print(np.shape(ucb),ucb)
        #ucb_norm = np.linalg.norm(ucb)
        
        if (max_ucb < ucb[0][0]):
            max_ucb = ucb[0][0]
            last_recommendation = x

            #print('yes')

    #print('recom: ' + str(last_recommendation) + '  for ucb: ' + str(max_ucb)),

    return last_recommendation
