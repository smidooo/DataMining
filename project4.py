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

def set_articles(articles):
    # method is called once at the beginning
    # all (80) article features are provided
    pass


def update(reward):
    # called after providing an article to the user
    # update the policy if necessary
    # reward:
    #  -1: recommended and displayed article didn't match
    #   0: article match, but user didn't click
    #   1: articles match, the user has clicked
    print('update running with reward ' + str(reward))


def recommend(time, user_features, choices):
    # called for every line in the logs
    # select one article from choices, return it (i.e. recommend for the user)
    # update method will be called with the result (i.e. click / no click)
    
    #print('recommend running...')
    #print(str(np.shape(time)))
    #print(str(np.shape(user_features)))
    #print(str(np.shape(choices)))

    return np.random.choice(choices)
