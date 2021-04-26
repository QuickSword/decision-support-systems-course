from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
n_HP_points_to_test = 100 #This parameter defines the number of HP points to be tested


def tune(X_train, y_train, param_test):

    gs = RandomizedSearchCV(
        estimator=model, param_distributions=param_test, 
        n_iter=n_HP_points_to_test,
        scoring='neg_log_loss',
        cv=3,
        refit=True,
        random_state=314,
        verbose=True)
    
    gs.fit(X_train, y_train)
    print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))
    return gs.best_params_