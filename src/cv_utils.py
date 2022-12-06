import numpy as np
from collections import Counter
import itertools


from sklearn.model_selection import ShuffleSplit


def AnomalyShuffleSplit(Xpos, Xneg, n_splits=5, test_size=.2, random_state=None):
    '''
    '''
    # generate splits on positive samples
    splits = ShuffleSplit(n_splits=n_splits, test_size=test_size,
                          random_state=random_state).split(Xpos)
        
    # generate splits which combines shuffled indices of positive samples
    # and always the same negative samples
    n, m = len(Xpos), len(Xneg)
    cv = [(train, np.concatenate([test, np.arange(n, n+m)], axis=0)) for train, test in splits]
    
    return cv


def RepeatedAnomalyShuffleSplit(Xpos, Xneg, n_splits=5, test_size=.2,
                                n_repeats=1, random_state=None):
    '''
    '''
    cv = [AnomalyShuffleSplit(Xpos=Xpos, Xneg=Xneg, n_splits=n_splits, test_size=test_size,
                              random_state=random_state) for i in range(n_repeats)]
    
    # flatten outer list
    #cv = list(itertools.chain(*cv))
    
    return cv


def CreateAnomalyData(Xpos, Xneg):
    '''
    '''
    # combine to full data set
    X = np.concatenate([Xpos, Xneg], axis=0)
    
    # generate lables according to sklearn standard (1=inlier, -1=outlier)
    y = np.concatenate([np.repeat(1.0, len(Xpos)), np.repeat(-1.0, len(Xneg))])
    
    return X, y
    

def dump_cv(cv, X=None, y=None):
    '''
    '''
    # assume sklearn interface for cv and yield generator values
    if type(cv) is not list:
        cv = list(cv.split(X, y)).copy()
        
    for i, (train_idx, test_idx) in enumerate(cv):
        print(f"---------cv iteration {i}----------")
        print("train")
        cnt = Counter(y[train_idx])
        print(cnt)


        print("test")
        cnt = Counter(y[test_idx])
        print(cnt)



#from debugger import tictoc
# keep training data, but replace original validation data with new test data
def replace_val_fold(cv, X, y, X_test, y_test):
    '''
    '''
    custom_cv = list(cv.split(X, y)).copy()
    
    # this is always the same, only train data changes according to CV
    test_fold = np.array(list(range(len(X), len(X)+len(X_test))))

    for i in range(len(custom_cv)):
        custom_cv[i] = (custom_cv[i][0], test_fold)
        
    return custom_cv, X.append(X_test), np.vstack([y, y_test])
