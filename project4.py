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

iteration_cnt = 0
nof_user_features = 6
nof_articles = None
article_features = None
last_recommendation = None
last_user_features = None
M = {}
b = {}


def set_articles(articles):
    # method is called once at the beginning
    # all (80) articles are provided (i.e. the ID and features)
    global nof_user_features, nof_articles, article_features, M, b

    # initialization or article features, M and b
    article_features = articles
    nof_articles = len(article_features)
    for a in articles:
        M[a] = np.identity(nof_user_features)
        b[a] = np.zeros(nof_user_features)


def update(reward):
    # called after providing an article to the user
    # update the policy if necessary
    # reward:
    #  -1: recommended and displayed article didn't match
    #   0: article match, but user didn't click
    #   1: articles match, the user has clicked

    global iteration_cnt, M, b, last_recommendation, last_user_features
    M_diff = np.multiply(last_user_features, last_user_features.T)
    M[last_recommendation] = np.add(M[last_recommendation], M_diff)
    b_diff = np.multiply(last_user_features, reward)
    b[last_recommendation] = np.add(b[last_recommendation], b_diff)
    iteration_cnt += 1


def recommend(time, user_features, choices):
    # called for every line in the logs
    # select one article from choices, return it (i.e. recommend for the user)
    # update method will be called with the result (i.e. click / no click)

    global iteration_cnt, M, last_recommendation, last_user_features
    C_ALPHA = 7
    last_user_features = np.matrix(user_features)
    max_ucb = None
    for c in choices:
        M_inv = np.linalg.inv(M[c])
        w = np.multiply(M_inv, b[c])
        proj = np.multiply(last_user_features.T, np.linalg.inv(M[c]))
        proj = np.multiply(proj, last_user_features)
        ucb_w = np.multiply(w.T, last_user_features)
        ucb_alpha = np.multiply(C_ALPHA, np.sqrt(proj))
        ucb = np.add(ucb_w, ucb_alpha)
        ucb_norm = np.linalg.norm(ucb)
        if (1 >= iteration_cnt):
            pass
            #print('testing ' + str(c))
            #print('  ## M_inv:     \n' + str(M_inv))
            #print('  ## w:         \n' + str(w))
            #print('  ## proj:      \n' + str(proj))
            #print('  ## ucb_w:     \n' + str(ucb_w))
            #print('  ## ucb_alpha: \n' + str(ucb_alpha))
            #print('  ## ucb:       \n' + str(ucb))
            #print('  ## ucb_norm:  ' + str(ucb_norm))
        if (np.isnan(ucb_norm)):
            print('ucb norm is NaN in iteration ' + str(iteration_cnt) + ' for choice ' + str(c))
        if (not np.isnan(ucb_norm) and ((max_ucb is None) or (max_ucb < ucb_norm))):
            max_ucb = ucb_norm
            last_recommendation = c

    if (10 >= iteration_cnt):
        print('recommendation: ' + str(last_recommendation))

    return np.random.choice(last_recommendation)
