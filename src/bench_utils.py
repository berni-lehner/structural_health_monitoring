""".
 
"""
__author__ = ("Bernhard Lehner <https://github.com/berni-lehner>")

import pandas as pd
import numpy as np

from sklearn import model_selection
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RepeatedKFold

def repeat_experiment(estimator, n_repeats, name):
    '''
    '''
    experiment = [(f"{name}_{i}", estimator) for i in range(n_repeats)]
    
    return experiment


def aabb_regression_benchmark(X, y,
                                  models,
                                  cv=None,
                                  scoring=None,
                                  random_state=None):
    '''
    '''
    all_cv_results = []
    for mdls in models:
        results = regression_benchmark(X=X, y=y,
                                            models=mdls,
                                            cv=cv,
                                            scoring=scoring,
                                            random_state=random_state)
        all_cv_results.append(results)
        
    all_cv_results = pd.concat(all_cv_results, ignore_index=True)
    
    return all_cv_results


def aabb_classification_benchmark(X, y,
                                  models,
                                  cv=None,
                                  scoring=None,
                                  random_state=None):
    '''
    '''
    all_cv_results = []
    for mdls in models:
        results = classification_benchmark(X=X, y=y,
                                            models=mdls,
                                            cv=cv,
                                            scoring=scoring,
                                            random_state=random_state)
        all_cv_results.append(results)
        
    all_cv_results = pd.concat(all_cv_results, ignore_index=True)
    
    return all_cv_results

    
def classification_benchmark(X, y,
                             models,
                             cv=None,
                             scoring=None,
                             groups=None,
                             random_state=None,
                             n_jobs=3):
    '''
    '''
    if cv is None:
        cv = StratifiedShuffleSplit(n_splits=8, test_size=0.1,
                                    random_state=random_state)
    if scoring is None:
        scoring = ['balanced_accuracy', 'accuracy', 'precision_macro',
                   'recall_macro', 'f1_macro']
        
    results = benchmark(X=X, y=y, models=models,
                        cv=cv,
                        scoring=scoring,
                        groups=groups,
                        random_state=random_state,
                        n_jobs=n_jobs)
    return results


def regression_benchmark(X, y,
                         models,
                         cv=None,
                         scoring=None,
                         groups=None,
                         random_state=None,
                         n_jobs=3):
    '''
    '''
    if cv is None:
        cv = RepeatedKFold(n_splits=5, n_repeats=3)

    if scoring is None:
        scoring = ['r2', 'neg_mean_squared_error',]
    
    results = benchmark(X=X, y=y, models=models,
                        cv=cv,
                        scoring=scoring,
                        groups=groups,
                        random_state=random_state,
                        n_jobs=n_jobs)
    return results
  

#@tictoc  
# IMPORTANT! if you get an error like this, the cv generator is exhausted, and it doesn't reset on its own.
# This causes the list of scored results to be empty, hence the error.
'''
File ~\Anaconda3\envs\ENVIRONMENT\lib\site-packages\sklearn\model_selection\_validation.py:1884, in _aggregate_score_dicts(scores)
   1857 def _aggregate_score_dicts(scores):
   1858     """Aggregate the list of dict to dict of np ndarray
   1859 
   1860     The aggregated output of _aggregate_score_dicts will be a list of dict
   (...)
   1878      'b': array([10, 2, 3, 10])}
   1879     """
   1880     return {
   1881         key: np.asarray([score[key] for score in scores])
   1882         if isinstance(scores[0][key], numbers.Number)
   1883         else [score[key] for score in scores]
-> 1884         for key in scores[0]
   1885     }

IndexError: list index out of range
'''
def benchmark(X, y,
              models,
              scoring,
              cv,
              groups=None,
              random_state=None,
              n_jobs=3):
    '''
    Evaluates the performance of multiple models using cross-validation.

    Parameters:
    - X: a NumPy array or Pandas DataFrame containing the training data.
    - y: a NumPy array or Pandas Series containing the target values.
    - models: a list of tuples containing the names and instances of the models to be evaluated.
    - scoring: a string or a list of strings representing the metric(s) to use for evaluation.
    - cv: an integer or a list of integers representing the number of folds or a list of lists of indices specifying the folds.
    - groups: a NumPy array or Pandas Series containing group labels for the samples.
    - random_state: the seed used by the random number generator.
    - n_jobs: the number of jobs to run in parallel.

    Returns:
    - A Pandas DataFrame containing the cross-validation results for all models.
    '''
    all_cv_results = []
               
    # for custom cv, you might want to have a separate cv for each model
    # TODO: sanity checks
    if type(cv[0]) is list:
        assert len(cv) == len(models), "list of cv list and models need to be equally long."
        
        for (name, model), cv_ in zip(models, cv):
            try:
                cv_results = model_selection.cross_validate(model, X, y,
                                                            cv=cv_, groups=groups,
                                                            scoring=scoring,
                                                            n_jobs=n_jobs)

            except ValueError as exc:
                print(exc)
                
            tmp = pd.DataFrame(cv_results)
            tmp['model'] = name
            all_cv_results.append(tmp)
    else:
        for name, model in models:
            try:
                cv_results = model_selection.cross_validate(model, X, y,
                                                            cv=cv, groups=groups,
                                                            scoring=scoring,
                                                            n_jobs=n_jobs)

            except ValueError as exc:
                print(exc)

            tmp = pd.DataFrame(cv_results)
            tmp['model'] = name
            all_cv_results.append(tmp)
    
    all_cv_results = pd.concat(all_cv_results, ignore_index=True)
    
    return all_cv_results


        
def extract_metrics(results: pd.DataFrame, metrics, sort=None):
    '''
    Extracts specified metrics from a DataFrame of results and returns them in a new DataFrame.
    Used foremost in conjunction with Visualization of the results.

    Parameters:
    - results: a Pandas DataFrame containing the results to be processed.
    - metrics: a list of strings representing the names of the metrics to extract (columns in results).
    - sort: a string representing the name of the column to sort the resulting DataFrame by.
        If not provided, the DataFrame will not be sorted.

    Returns:
    - A new DataFrame containing only the specified metrics and sorted (if specified).
    '''
    results = pd.melt(results, id_vars=['model'],
                        var_name='metrics', value_name='values')

    results = results.loc[results['metrics'].isin(metrics)]

    if sort is not None:
        results = results.sort_values(by=sort)

    return results
