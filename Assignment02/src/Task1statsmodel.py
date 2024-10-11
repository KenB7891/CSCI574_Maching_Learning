import sys
import numpy as np
import statsmodels.api as sm


def task1_statsmodel(y, X):
    """Given the asked for features X (with a dummy intercept constant
    already added), and the target regression values y, fit a
    statsmodel OLS (ordinary least squares) regression model to the
    data and return the fitted model along with important fit parameters.

    Parameters
    ----------
    y - The expected set of regression targets in order to get the expected
        fitted regression model for task 1
    X - The expected set of features to train with, with an already added
        dummy intercept constant, in order to get the expected
        fitted regression model for Task 1

    Returns
    -------
    model, intercept, slope, mse, rmse, r2score - Returns a tuple of the fitted
        model, along with some parameters from the fit, including the slope and
        intercept, mse, rmse and r2 score

    Tests
    -----
    # these tests assume X and y are already defined in envrionment where
    # the doctests are called, and even more that the particular dataframe and
    # expected X input features and y regression targets are being used that
    # will produce the expected model and results from fitting the model
    >>> from AssgUtils import isclose
    >>> model, params, rsquared = task1_statsmodel(y, X)

    # the params of a statsmodel OLS model contains the [intercept, slope1, slope2...]
    >>> isclose(params[0], 0.37578175021210747)
    True
    >>> isclose(params[1], 0.3354845860060065)
    True
    >>> isclose(rsquared, 0.5008050204985712)
    True

    """
    # your code for Task 1 statsmodel regression model
    # goes here
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)

    # should remove these once you are actually creating them correctly
    params = model.params
    rsquared = model.rsquared

    return model, params, rsquared


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
