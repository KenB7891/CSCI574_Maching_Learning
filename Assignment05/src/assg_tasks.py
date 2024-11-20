import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier


def task_1_load_data():
    """Task 1 loads data from the MagicTelescope OpenML data repository.  You
    are required to use the fetch_openml() function to retrieve this
    dataset.  In addition you will need to do some work to encode the
    labels and reshape the data.  The assignment expects that the X
    input fetures are returend with 19020 samples of 11 features, and that
    the y labels are encoded correctly for binary classification.

    Params
    ------

    Returns
    -------
    X - A (19020,10) shaped Pandas dataframe of the numeric features from the
        MagicTelescope dataset.
    y - The binary labels for the dataset, a Numpy ndarray vector of shape
        (19020,) encoded as integers where the `g` class has been encoded
        as the positive 1 class and the `h` class as the negative 0 class.
    """
    # your code to load the MagicTelescope data and format it as asked goes here
    data = fetch_openml('MagicTelescope', version=1).frame
    labels = data['class:']
    ordered_categories = [['h', 'g']]
    encoder = OrdinalEncoder(categories=ordered_categories)
    y = encoder.fit_transform(labels.to_numpy().reshape(-1, 1)).flatten()
    X = data.drop(['class:'], axis='columns')

    return X, y


def train_val_test_split(data, target, train_size=10000, val_size=4510, random_state=42):
    """Task 2, create a utility function to perform a 3 way split of some
    data/target array like types into training data, validation data and
    test data.  The absolute size of the training and validation data
    in terms of the number of samples is required for this function.  
    All remaining data/targets will be put into the final test data set. 
    This function assumes that the input data is large enough to perform 
    the indicated 3 way split, no error checking is done or required.

    The random_state should be set for both calls to sklearns train_test_split().
    If not the split will not be reproducible as the behavior in sklearn is to
    use a different seed each time if not specified.

    Params
    ------
    data - An array or dataframe like entity that will be split 3 ways.
    target - An array or series like entity that has corresponding targets for
      the data and should be split in exactly the same way as the data.
    train_size - The number of samples that should end up in the training
       data set and in the training target labels returned
    val_size - The number of samples that should end up in the validation data
       set and in the validation target labels
    random_state - A value to set the random seed to before performing
       split, defaults to 42 if not specified.  Allows for reproducable splits
       to be done by this function.

    Returns
    -------
    (X_train, y_train, X_val, y_val, X_test, y_test) - Returns a tuple in
       the shown order of the 3-way split of data and targets into train,
       validation and test sets.
    """
    
    # your implementation goes here
    X_train, X_temp, y_train, y_temp = train_test_split(data, target, train_size=train_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=val_size, random_state=random_state)

    return (X_train, y_train, X_val, y_val, X_test, y_test)


def task_3_voting_ensemble(X, y, voting='hard'):
    """In this task function you are to create 5 different classifiers 
      1. 'knn' k-nearest neighbors 
      2. 'dt' decision tree
      3. 'lr' logistic regression
      4. 'svc' support vector classifier
      5. 'mlp' multi-layer perceptron
    Use the name shown in the list of tuples to create a voting ensemble
    from the named estimators.  The voting ensemble should use either hard or
    soft voting as specified by the third parameter to this function.  The
    voting ensemble should be fit on the given data and labels and the fitted
    ensemble returned from this function.
    
    Paramters
    ---------
    X - Data / features to train and fit the ensemble model with.
    y - Labels / targets for a binary classification task expected by this task
    voting - A string that should specifiy either 'hard' or 'soft' to be used
       when creating the voting ensemble that is returned from this function
      
    Returns
    -------
    voting_ensemble - Creates and returns a voting ensemble of the noted
      classifiers that has been fit and trained on the training data given
      to this function.
    """
    # task 3 implementation goes here
    knn_clf = KNeighborsClassifier()
    dt_clf = DecisionTreeClassifier()
    lr_clf = LogisticRegression(solver='sag', max_iter=10000)
    svc_clf = SVC(probability=True)
    mlp_clf = MLPClassifier()

    voting_clf = VotingClassifier(
        estimators=[('knn', knn_clf), ('dt', dt_clf), ('lr', lr_clf), ('svc', svc_clf), ('mlp', mlp_clf)],
        voting=voting
    )

    voting_clf.fit(X, y)
    
    return voting_clf


def task_4_bag_of_trees_ensemble(X, y):
    """In this task create, fit and return a bagging ensemble.  You can
    use either a random forest or an extra trees classifier.  Try and tune
    the meta-parameters to get the best performance accuracy you can on
    your classifier.
    
    Paramters
    ---------
    X - Data / features to train and fit the ensemble model with.
    y - Labels / targets for a binary classification task expected by this task

    Returns
    -------
    bagging_ensemble - Creates, trains and returns a bag of trees ensemble for
      the binary classification task.
    """
    # task 4 implementation goes here
    estimators = 2000
    max_node = 60
    bagging_clf = RandomForestClassifier(n_estimators=estimators, max_leaf_nodes=max_node, n_jobs=-1)
    bagging_clf.fit(X, y)
    return bagging_clf


def create_stacked_data(voting_ensemble, X):
    """Given a VotingClassifier voting ensemble, access each estimator in
    the ensemble and generate its probability predictions for the given
    input data X.  Gather all of the prediction outputs and stack into
    a array to be returned.  In this assignment we are performing binary
    classification, so if X has 5000 samples, each estimator should return
    an array shaped (5000, 2) with the two probability estimates for class
    0 and 1 respectively.  If the voting ensemble has 5 base estimators
    in it, then we would expect a (5000, 10) shaped stacked array to be
    returned.

    Parameters
    ----------
    voting_ensemble - Should be a VotingClassifer from scikit-learn that has
      already been trained and fitted.  We will use the estimators in the
      voting ensemble as base estimators for a stack ensemble.
    X - The data that we should generate predictions for from each of the
      base estimators and that will be stacked together as the result.  

    Returns
    -------
    stacked_array - Return result of horizontally stacking the arrays.  If The
       individual outputs from the base estimators are shaped (5000, 2),
       and there are 5 estimators in the voting ensemble to stack, then the
       resulting stacked_array will be of shape (5000, 10), and
       the columns 0 and 1 will be from the first estimator in the voting
       ensemble, columns 1 and 2 from the next, and so on.
    """
    # your implementation goes here
    estimators = voting_ensemble.named_estimators_
    prob_lists = []
    for name, classifier in estimators.items():
        prob_predictions = classifier.predict_proba(X)
        prob_lists.append(prob_predictions)

    stacked_array = np.hstack(prob_lists)
    return stacked_array


def task_6_stacked_ensemble(X_train, y_train, X_val, y_val):
    """Created 5 base estimators (reusing the task 3 method to create
    a voting ensemble) and then stack and blend their predictions by
    creating a SVC estimator that is trained on the outputs of the 5 base
    estimators to make final predictions.

    You are required to reuse your task_3_voting_ensemble() and your
    stack_data() methods in the task 6 implementation. You need to
    train the base estimators on the training data given, and then
    generate and stack a new set of data for training and fitting
    using the validation data set.
    
    Paramters
    ---------
    X_train, y_train - The training data set, should be used to fit the base
      estimators of the stack.
    X_val, y_val - The validation data set, should be used here to create
      a new set of training data from the base estimators that will be used
      to train the final blending estimator.
      
    Returns
    -------
    blending_estimator - The trained blender estimator that blends the stacked
      output of the base estimators from the voting ensemble
    voting_ensemble - The set of base estimators used for this stacked ensemble
    """
    # task 6 implementation goes here
    voting_ensemble = task_3_voting_ensemble(X_train, y_train, 'soft')

    blended_data = create_stacked_data(voting_ensemble, X_val)

    blending_estimator = SVC(probability=True)
    blending_estimator.fit(blended_data, y_val)
    
    return blending_estimator, voting_ensemble