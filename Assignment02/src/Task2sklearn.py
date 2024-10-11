import sys
import numpy as np
from sklearn.linear_model import LogisticRegression


def task2_sklearn(X, y):
    """
    Given the asked for features X and the target regression
    values y, fit a sklearn logistic regression model to perform
    a classification task.

    Parameters
    ----------
    X - The expected set of features to train with in order to get the expected
        fitted regression model for Task 1
    y - The expected set of regression targets in order to get the expected
        fitted regression model for task 1

    Returns
    -------
    model, intercept, slopes, accuracy - Returns a tuple of the fitted
        model, along with some parameters from the fit, including the slopes and
        intercept, and accuracy on the training data

    Tests
    -----
    # these tests assume X and y are already defined in envrionment where
    # the doctests are called, and even more that the particular dataframe and
    # expected X input features and y regression targets are being used that
    # will produce the expected model and results from fitting the model
    >>> from AssgUtils import isclose
    >>> model, intercept, slopes, accuracy = task2_sklearn(X, y)
    >>> isclose(intercept[0], 186.590648)
    True
    >>> isclose(slopes[0][0], -0.320885)
    True
    >>> isclose(slopes[0][1], -0.183120)
    True
    >>> isclose(accuracy, 0.863388)
    True
    """
    # your code for Task 2 scikit-learn classification model
    # goes here
    model = LogisticRegression(solver='lbfgs', C=500.0)
    model.fit(X,y)

    predictions = model.predict(X)
    score = model.score(X, y)
    
    # should remove these once you are actually creating them correctly
    intercept = model.intercept_
    slopes = model.coef_
    accuracy = score

    return model, intercept, slopes, accuracy


if __name__ == "__main__":
    import doctest
    import pandas as pd
    # the doctests here expect that X and y are alread defined
    # in the environment, and are the specific features X and
    # regression targets y being tested
    X = pd.read_pickle('data/classification_features.pkl')
    y = np.load('data/classification_labels.npy')

    # we execute doctests and return the number of failing tests
    # to exit, thus if 1 or more fail, we return non zero exit code
    failure_count, test_count = doctest.testmod()
    sys.exit(failure_count)
