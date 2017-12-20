import numpy as np

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

nof_user_features = 6
nof_articles = None
article_features = None
last_recommendation = None
last_user_features = None
M = {}
b = {}

C_ALPHA = 12 

def set_articles(articles):
    # method is called once at the beginning
    # all (80) articles are provided (i.e. the ID and features)
    global nof_user_features, nof_articles, article_features, M, b

    # initialization or article features, M and b
    article_features = articles
    nof_articles = len(article_features)
    for a in articles:
        M[a] = np.identity(nof_user_features).reshape(6,6)
        b[a] = np.zeros(nof_user_features).reshape(6,1)


def update(reward):
    # called after providing an article to the user
    # update the policy if necessary
    # reward:
    #  -1: recommended and displayed article didn't match
    #   0: article match, but user didn't click
    #   1: articles match, the user has clicked
    #print('update')
    global M, b, last_recommendation, last_user_features

    M_diff = np.multiply(last_user_features.reshape(6,1), last_user_features.reshape(6,1).T)
    M[last_recommendation] = M[last_recommendation] + (M_diff)
    b[last_recommendation] = b[last_recommendation] + (last_user_features*reward)
    
    #print('Mdiff', np.shape(M_diff), M_diff)

    #print('M', last_recommendation, np.shape(M[last_recommendation]), M[last_recommendation])
    #print('b',last_recommendation,np.shape(b[last_recommendation]), b[last_recommendation])
    #print(np.shape(last_user_features), last_user_features)


def recommend(time, user_features, choices):
    # called for every line in the logs
    # select one article from choices, return it (i.e. recommend for the user)
    # update method will be called with the result (i.e. click / no click)

    #print ('recommend')
    global M, last_recommendation, last_user_features
    last_user_features = np.array(user_features).reshape(6,1)
    max_ucb = 0

    #print('M',np.shape(M))
    #print('b',np.shape(b))
    #print('last_user_features', np.shape(last_user_features))


    #for all x actoins in action set A_t (choices)
    for x in choices:
        w = np.dot(np.linalg.inv(M[x]), b[x]).reshape(6,1)
        #print('w', np.shape(w))
        proj1 = np.dot(last_user_features.T, np.linalg.inv(M[x])).reshape(1,6)
        #print('pro1', np.shape(proj1))
        proj = np.dot(proj1, last_user_features)
        #print('proj', np.shape(proj))

        ucb = np.dot(w.T, last_user_features) + (C_ALPHA * np.sqrt(proj))

        #print(np.shape(ucb))
        #ucb_norm = np.linalg.norm(ucb)
        
        if (max_ucb < ucb):
            max_ucb = ucb
            last_recommendation = x

            #print('yes')

    print('recom: ' + str(last_recommendation))

    return last_recommendation