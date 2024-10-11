import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def task1_load_data():
    """Task 1 is to load and return the data in the correct shape
    from the assignment data file.  You need to use the correct
    relative path name to load the data here.  The only tricky
    part is that the x input data needs to be a 2-d column array,
    (e.g. shape (100,1), 100 sample rows with 1 column).

    Note: This function is reused in most of the assignment tests to
    get the data that you should always be using when fitting/training
    models.

    Params
    ------

    Returns
    -------
    x, y - Returns the 1 feature x inputs we will use in this assignment.
       This should be an (100,1) shaped 2-d numpy array.  y contains the
       regression targets for the secret noisy function.  This should be
       a numpy array but a 1-d vector of 100 items.
    """
    # task 1 code to load the data and format it as needed goes here
    # make sure you return the expected values
    df_data = pd.read_csv('../data/assg-03-data.csv')
    x = df_data['x'].values.reshape(100,1)
    y = df_data['y'].values
    return x, y


def task2_underfit_model(x, y):
    """Create, train and return the described underfit model using a degree
    2 polynomial for task 1.  You should create a pipeline to
    turn the features into a degree 2 polynomial that feeds into a linear
    regression model.  Then fit the model.  This function returns
    the trained underfit model you create for task 2.

    Params
    ------
    x - The input features, we expect a numpy array with 100 samples and only
       1 input feature.
    y - The output regression target labels to fit the model to.

    Returns
    -------
    model - Returns the trained underfit model/pipeline.
    """
    # task 2 code to create underfit model pipeline goes here
    # make sure you return the expected model
    polynomial_regression = Pipeline([
        ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
        ('lin_reg', LinearRegression()),
    ])

    polynomial_regression.fit(x, y)
    
    return polynomial_regression


def learning_curve_errors(model, X, y):
    """Calculate learning curve errors obtained with training the given scikit-learn model
    with progressively larger amounts of the training data X.
    
    This function splits the data into 80% used to train models, and 20% for final
    validation.  The learning curves are estimated by then using the training data
    to train a model on just 1 data point, then on 2, etc. up to all of the training
    data.  The performance of the model is then evaluated both on the same data it was
    just trained with, and on the held back 20% of data used for validation. 
    This function collects the RMSE errors on train and validation and returns
    them as the result.
    
    Parameters
    ----------
    model - A scikit-learn estimator model to be trained and evaluated.
    X - The input training data
    y - The target labels for training

    Returns
    -------
    train_errors, test_errors - Returns a list of the root mean squared errors
       RMSE of the performance of trained models on the original data trained with,
       and on 20% of the data held back for validation.  The returned arrays are
       the train/test performance on models created with 1 data point, then
       2, etc. up to the 80% of data we use for training models.

    Tests
    -----
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> np.random.seed(42)
    >>> x_doctest = np.linspace(-5.0, 5.0, 100)
    >>> y_doctest = 2.5 * x_doctest + 1.0 + np.random.randn(100) * 1.0
    >>> x_doctest = x_doctest.reshape(-1, 1)
    >>> model = LinearRegression()
    >>> train_errors, test_errors = learning_curve_errors(model, x_doctest, y_doctest)
    >>> len(train_errors)
    79
    >>> len(test_errors)
    79
    >>> train_errors[:5]
    [0.0, 1.3710242980056706e-15, 0.008226499792554355, 0.6060453617587794, 0.5519805145155924]
    >>> test_errors[:5]
    [6.807111779574341, 2.8618559336108844, 2.8588563844249277, 1.8483844530375642, 1.9383035157683932]
    >>> max(train_errors)
    1.0929159457839905
    >>> max(test_errors)
    6.807111779574341
    """
    # task 3 code to generate learning curve train and test errors goes here
    # make sure you return the expected errors
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    train_errors = []
    test_errors = []

    for i in range(1, len(X_train)):
        model.fit(X_train[:i], y_train[:i])
        
        y_train_predict = model.predict(X_train[:i])
        y_test_predict = model.predict(X_test)
        
        rmse_train = root_mean_squared_error(y_train[:i], y_train_predict)
        rmse_test = root_mean_squared_error(y_test, y_test_predict)
        train_errors.append(rmse_train)
        test_errors.append(rmse_test)

    return train_errors, test_errors


def task4_overfit_model(x, y):
    """Create, train and return the described overfit model using a degree
    100 polynomial for task 4.  You should create a pipeline to
    turn the features into a degree 100 polynomial that feeds into a linear
    regression model.  Then fit the model.  This function returns
    the trained overfit model you create for task 4.

    Params
    ------
    x - The input features, we expect a numpy array with 100 samples and only
       1 input feature.
    y - The output regression target labels to fit the model to.

    Returns
    -------
    model - Returns the trained overfit model/pipeline.
    """
    # task 4 code to create overfit model pipeline goes here
    # make sure you return the expected model
    polynomial_regression = Pipeline([
        ('poly_features', PolynomialFeatures(degree=100, include_bias=False)),
        ('lin_reg', LinearRegression()),
    ])

    polynomial_regression.fit(x, y)
    
    return polynomial_regression

def task5_lasso_model(x, y):
    """Create, train and return the described Lasso regularization model
    using a degree 100 polynomial for task 5.  You should create a pipeline to
    turn the features into a degree 100 polynomial that feeds into a Lasso
    regularization regression model.  Then fit the model.  This function returns
    the trained Lasso regularization model you create for task 5.

    Params
    ------
    x - The input features, we expect a numpy array with 100 samples and only
       1 input feature.
    y - The output regression target labels to fit the model to.

    Returns
    -------
    model - Returns the trained Lasso regularization model/pipeline.
    """
    # task 5 code to create Lasso regularization model pipeline goes here
    # make sure you return the expected model
    lasso_reg = Pipeline([
        ('poly_features', PolynomialFeatures(degree=100, include_bias=False)),
        ('lasso_reg', Lasso(alpha=0.0005, max_iter=500000)),
    ])

    lasso_reg.fit(x, y)
    return lasso_reg

def task6_ridge_model(x, y):
    """Create, train and return the described Ridge regularization model
    using a degree 100 polynomial for task 5.  You should create a pipeline to
    turn the features into a degree 100 polynomial that feeds into a Ridge
    regularization regression model.  Then fit the model.  This function returns
    the trained Ridge regularization model you create for task 6.

    Params
    ------
    x - The input features, we expect a numpy array with 100 samples and only
       1 input feature.
    y - The output regression target labels to fit the model to.

    Returns
    -------
    model - Returns the trained Ridge regularization model/pipeline.
    """
    # task 6 code to create Ridge regularization model pipeline goes here
    # make sure you return the expected model
    ridge_reg = Pipeline([
        ('poly_features', PolynomialFeatures(degree=100, include_bias=False)),
        ('ridge_reg', Ridge(alpha=0.0005, max_iter=500000)),
    ])

    ridge_reg.fit(x, y)
    return ridge_reg