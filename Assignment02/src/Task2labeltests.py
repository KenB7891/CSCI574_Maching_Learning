import sys
import numpy as np


def task2_label_tests(y):
    """
    At the start of task 2, you first need to encode the RainTommorow as
    a categorical variable, and correctly map No to the 0 encoding and 
    Yes to the 1 encoding.  The encoded labels are passed in and tested by
    this function.

    Parameters
    ----------
    y - The expected set of classification labels for task 2, correctly encoded
        as a categorical variable and mapped as required.

    Returns
    -------
    ndim, shape, num_no, num_yes - The array should be 1 dimensional with 366 values, and
       if encoded correctly, there are 300 No and 66 Yes.  These are extractred and
       returned so we can test.

    Tests
    -----
    # these tests assume y is already defined in envrionment where
    # the doctests are called, and that it contains the correctly
    # encoded categorical labels for Task 2
    >>> ndim, shape, num_no, num_yes = task2_label_tests(y)
    >>> ndim
    1
    >>> shape
    (366,)
    >>> num_no
    300
    >>> num_yes
    66
    """
    ndim = y.ndim
    shape = y.shape
    num_no = sum(y == 0)
    num_yes = sum(y == 1)
    return ndim, shape, num_no, num_yes


if __name__ == "__main__":
    import doctest
    # we execute doctests and return the number of failing tests
    # to exit, thus if 1 or more fail, we return non zero exit code
    y = np.array([1, 2, 3, 4])
    failure_count, test_count = doctest.testmod()
    sys.exit(failure_count)
