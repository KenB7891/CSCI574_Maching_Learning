import sys
import numpy as np


def task2_feature_tests(X):
    """
    Next in task 2, you need to encode extract the Sunshine and Pressure3pm features,
    and perform some data cleaning to impute some missing features.  The feature
    dataframe is passed in and information extracted from it to ensure that
    you have imputed the values and created the input features as expected for
    task 2.

    Parameters
    ----------
    X - The expected set of input features for task 2, correctly cleaned
        and missing values imputed.

    Returns
    -------
    ndim, shape, columns, na_sum, description - The feature table should be 2 dimensional
       but also with 366 values.  There should not be any missing values, and the mean and other
       numeric summary descriptions should match if missing values were imputed with mean as asked

    Tests
    -----
    # these tests assume y is already defined in envrionment where
    # the doctests are called, and that it contains the correctly
    # encoded categorical labels for Task 2
    >>> ndim, shape, columns, na_sum, description = task2_feature_tests(X)
    >>> ndim
    2
    >>> shape
    (366, 2)
    >>> columns
    Index(['Sunshine', 'Pressure3pm'], dtype='object')
    >>> na_sum
    Sunshine       0
    Pressure3pm    0
    dtype: int64
    >>> description
             Sunshine  Pressure3pm
    count  366.000000   366.000000
    mean     7.909366  1016.810383
    std      3.467180     6.469422
    min      0.000000   996.800000
    25%      6.000000  1012.800000
    50%      8.600000  1017.400000
    75%     10.500000  1021.475000
    max     13.600000  1033.200000
    """
    ndim = X.ndim
    shape = X.shape
    columns = X.columns
    na_sum = X.isna().sum()
    description = X.describe()
    
    return ndim, shape, columns, na_sum, description


if __name__ == "__main__":
    import doctest
    # we execute doctests and return the number of failing tests
    # to exit, thus if 1 or more fail, we return non zero exit code
    X = np.array([1, 2, 3, 4]).reshape((2, 2))
    failure_count, test_count = doctest.testmod()
    sys.exit(failure_count)
