import doctest
import math


def run_doctests(func, globs):
    """This function is meant to run inside of an iPython notebook.  You
    pass in a function that has doctests defined in the function docstring.
    This function finds all of the doctests in the function documentation
    and runs them.

    Parameters
    ----------
    func - A python defined function that contains doctests in its function
      docstring

    Returns
    -------
    func - The passed in function is returned, so that calls can be chained
      when calling this function.
    """
    #globs = copy.copy(globals())
    globs.update({func.__name__: func})
    doctest.run_docstring_examples(func, globs, verbose=True, name=func.__name__)
    return func


def isclose(actual, expected, rel_tol=1e-04):
    """Wraper around the math.isclose() function, where
    we use a lower default relative tolerance.  But more
    importantly, on failure we display the actual and expected
    values we were given.
    """
    if math.isclose(actual, expected, rel_tol=rel_tol):
        return True
    else:
        msg = "False: expected %f but actual value %f" % (expected, actual)
        return msg
