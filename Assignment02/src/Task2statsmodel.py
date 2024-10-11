import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

def prob_to_category(pred_probs):
    """We expect a samples X 10 shaped numpy array of prediction probabilities for the
    10 categories.  We return a samples x 1 shaped vector of the corresponding category label
    """
    return np.argmax(pred_probs, axis=1)

def score(pred, y):
    """Given a vector of predictions and a a vector of correct labels y,
    determine the score or prediction accuracy.
    """
    n = y.shape[0] # number of samples
    correct_labels = (pred == y)
    
    # return the accuracy ration, number correct / number of samples
    return np.sum(correct_labels) / n

def task2_statsmodel(y, X):
    """Given the asked for features X (with a dummy intercept constant
    already added), and the target regression values y, fit a
    statsmodel Logit  (logitsic regression) regression model to the
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
    model, params, accuracy - Returns a tuple of the fitted
        model, along with some parameters from the fit, including the
        fitted model parameters and the final model accuracy

    Tests
    -----
    # these tests assume X and y are already defined in envrionment where
    # the doctests are called, and even more that the particular dataframe and
    # expected X input features and y regression targets are being used that
    # will produce the expected model and results from fitting the model
    >>> from AssgUtils import isclose
    >>> model, params, accuracy = task2_statsmodel(y, X)
    Optimization terminated successfully.
             Current function value: 0.324586
             Iterations 7

    # the params of a statsmodel OLS model contains the [intercept, slope1, slope2...]
    >>> isclose(params['const'], 186.59040174670466)
    True
    >>> isclose(params['Sunshine'], -0.3208828310913976)
    True
    >>> isclose(params['Pressure3pm'], -0.18311988277396155)
    True
    >>> isclose(accuracy, 0.8633879781420765)
    True

    """
    # your code for Task 2 statsmodel classification model
    # goes here
    X = sm.add_constant(X)
    logit = sm.MNLogit(y, X)

    model = logit.fit()
    predict_probs = model.predict(X)
    
    # should remove these once you are actually creating them correctly
    d = {'const': model.params[0].iloc[0], 'Sunshine': model.params[0].iloc[1], 'Pressure3pm': model.params[0].iloc[2]}
    params = pd.Series(data=d, index=['const', 'Sunshine', 'Pressure3pm'])
    predictions = prob_to_category(predict_probs)
    accuracy = score(predictions, y)
    
    return model, params, accuracy


if __name__ == "__main__":
    import doctest
    # the doctests here expect that X and y are alread defined
    # in the environment, and are the specific features X and
    # regression targets y being tested
    X = pd.read_pickle('data/classification_features.pkl')
    y = np.load('data/classification_labels.npy')

    # we execute doctests and return the number of failing tests
    # to exit, thus if 1 or more fail, we return non zero exit code
    failure_count, test_count = doctest.testmod()
    sys.exit(failure_count)
