import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def task1_sklearn(X, y):
    """
    Given the asked for features X and the target regression
    values y, fit a sklearn linear regression model to the data
    and return the fitted model.

    Parameters
    ----------
    X - The expected set of features to train with in order to get the expected
        fitted regression model for Task 1
    y - The expected set of regression targets in order to get the expected
        fitted regression model for task 1

    Returns
    -------
    model, intercept, slope, mse, rmse, rsquared - Returns a tuple of the fitted
        model, along with some parameters from the fit, including the slope and
        intercept, mse, rmse and r2 score

    Tests
    -----
    # these tests assume X and y are already defined in envrionment where
    # the doctests are called, and even more that the particular dataframe and
    # expected X input features and y regression targets are being used that
    # will produce the expected model and results from fitting the model
    >>> from AssgUtils import isclose
    >>> model, intercept, slopes, mse, rmse, rsquared = task1_sklearn(X, y)
    >>> isclose(intercept, 0.37578175021210747)
    True
    >>> isclose(slopes[0], 0.3354845860060065)
    True
    >>> isclose(mse, 3.5473465427798607)
    True
    >>> isclose(rmse, 1.8834400820784984)
    True
    >>> isclose(rsquared, 0.5008050204985712)
    True
    """
    # your code for Task 1 scikit-learn regression model
    # goes here
    model = LinearRegression().fit(X,y)

    # should remove these once you are actually creating them correctly
    intercept = model.intercept_
    slopes = model.coef_

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    rsquared = model.score(X,y)
    
    return model, intercept, slopes, mse, rmse, rsquared


if __name__ == "__main__":
    import doctest
    # the doctests here expect that X and y are alread defined
    # in the global environment, and are the specific features X and
    # regression targets y being tested
    X = np.load('data/regression_features.npy')
    y = np.load('data/regression_labels.npy')

    # we execute doctests and return the number of failing tests
    # to exit, thus if 1 or more fail, we return non zero exit code
    failure_count, test_count = doctest.testmod()
    sys.exit(failure_count)
