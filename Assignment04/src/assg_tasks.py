import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def task_1_1_load_data():
    """Task 1 loads data from the `../data/assg-04-data1.csv` file.  This
    function should load the two features into a Pandas dataframe, and the
    y labels should be returned as a separate NumPy array.  Make sure that
    you correctly label the feature columns 'x_1' and 'x_2' respectively.
    This data file does not have a special row that specifies the feature /
    target labels, you need to correctly specify these when or after loading
    the data.

    Params
    ------

    Returns
    -------
    X - Returns the 2 features as a pandas dataframe, labeled 'x_1' and
        'x_2' respectively.  The shape should be (51,2) for this dataframe. 
    y - Returns the binary labels as a regular (51,) shaped Numpy 1-D vector.
    """
    # need to actually load the correct data file and return the training features
    # as X, a pandas data frame, and y, a numpy array of the binary labels
    columns = ['x_1', 'x_2', 'label']
    data = pd.read_csv('../data/assg-04-data1.csv',header=None, names=columns)
    X = data[['x_1', 'x_2']]
    y = np.array(data['label'])

    return X, y


def task_1_2_linear_svm_classifier(X, y, C=1.0):
    """Task 1 part 2, create a linear classifier.  The features are expected
    to be passed in as a pandas dataframe `X`, and the binary labels in a numpy
    array `y`.  The `C` parameter controls the amount of regularization for
    a support vector classifier SVC as we discussed in class.  This parameter
    should be passed into and used when creating the SVC instance.

    You are expected to create and return a fitted pipeline from this function
    on the given data using the given C parameter.  The pipeline should first
    use a standard scalar to scale the input feature data.  Then a SVC
    classifier should be created, using the C parameter and a `linear` kernel.
    The resulting fitted pipeline / model should be returned from this function.

    Params
    ------
    X - A dataframe loaded with the task 1 data containing two features x_1 and
        x_2
    y - The binary classification labels for task 1 as a numpy array of 0 / 1
        values
    C - The regularization parameter to use for the SVC pipeline that will be
        created, defaults to C=1.0

    Returns
    -------
    pipeline - Returns a sklearn pipeline that contains a standard scalar that
      feeds into a SVC classifier using a linear kernel and the indicated C
      regularization parameter.
    """
    # create a pipeline to scale the data and fit a SVC classifier using a 
    # linear kernel.  Make sure you use the passed in C parameter when
    # creating your model
    svm_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='linear', C=C))
    ])

    svm_clf.fit(X, y)
    return svm_clf

def gaussian_kernel(xi, xj, sigma):
    """ Define gaussian kernel function.  Given two separate points xi and xj, calculate
    the gaussian kernel similarity.  The sigma parameter controls the width of the gaussian kernel
    similarity.
    
    Paramters
    ---------
    xi, xj - Numpy vectors of of 2 n-dimensional points.  Both vectors must be (n,) shaped (of same size
      and shape).
    sigma - meta parameter to control the width of the gaussian kernel (e.g. the standard deviation of the
      gaussian distribution being used.
      
    Returns
    -------
    K_gaussian - returns the gaussian kernel similarity measure of the distance between the 2 points.
    """
    # implement the described gaussian kernel / similarity function here
    norm = 0
    for i in range(len(xi)):
        norm += ((xi[i] - xj[i]) ** 2)

    K_gaussian = math.exp(-1 * (norm / (2 * (sigma ** 2))))

    
    return K_gaussian


def task_3_1_load_data():
    """Task 1 loads data from the `../data/assg-04-data2.csv` file.  This
    function should load the two features into a Pandas dataframe, and the
    y labels should be returned as a separate NumPy array.  Make sure that
    you correctly label the feature columns 'x_1' and 'x_2' respectively.
    This data file does not have a special row that specifies the feature /
    target labels, you need to correctly specify these when or after loading
    the data.

    Params
    ------

    Returns
    -------
    X - Returns the 2 features as a pandas dataframe, labeled 'x_1' and
        'x_2' respectively.  The shape should be (51,2) for this dataframe. 
    y - Returns the binary labels as a regular (51,) shaped Numpy 1-D vector.
    """
    # need to actually load the correct data file and return the training features
    # as X, a pandas data frame, and y, a numpy array of the binary labels
    columns = ['x_1', 'x_2', 'label']
    data = pd.read_csv('../data/assg-04-data2.csv',header=None, names=columns)
    X = data[['x_1', 'x_2']]
    y = np.array(data['label'])
    return X, y


def task_3_2_rbf_svm_classifier(X, y, kernel='rbf', C=1.0, gamma=8.0):
    """Task 3 part 2, create a SVC classifier using a nonlinear `rbf` kernel.
    The features are expected to be passed in as a pandas dataframe `X`,
    and the binary labels in a numpy array `y`.  The `C` parameter controls
    the amount of regularization for a support vector classifier SVC as
    we discussed in class.  Likewise the `gamma` parameter controls the
    shape of the `rbf` kernel used in this function.  These parameters
    should be passed into and used when creating the SVC instance.

    You are expected to create and return a fitted pipeline from this function
    on the given data using the given C parameter.  The pipeline should first
    use a standard scalar to scale the input feature data.  Then a SVC
    classifier should be created, using the C parameter and a `linear` kernel.
    The resulting fitted pipeline / model should be returned from this function.

    Params
    ------
    X - A dataframe loaded with the task 1 data containing two features x_1 and
        x_2
    y - The binary classification labels for task 1 as a numpy array of 0 / 1
        values
    C - The regularization parameter to use for the SVC pipeline that will be
        created, defaults to C=1.0
    gamma - The "spread" of the radial basis kernel function to use.

    Returns
    -------
    pipeline - Returns a sklearn pipeline that contains a standard scalar that
      feeds into a SVC classifier using a nonlinear rbf kernel and the indicated C
      and gamma regularization parameters.
    """
    # create a pipeline to scale the given data and then fit a SVC support
    # vector machine classifier to the data.  You need to use the
    # specified kernel for your SVC classifier, as well as the given
    # C and gamma parameters.

    rbg_svm_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel=kernel, gamma=gamma, C=C))
    ])

    rbg_svm_clf.fit(X,y)
    
    return rbg_svm_clf

def plot_svm_decision_boundary(model, X, y, 
                           class_0_label = 'Negative Class (0)', class_1_label = 'Positive Class (1)',
                           feature_1_label = r'$x_1$', feature_2_label = r'$x_2$',
                           remove_bias=False):
    '''Plot the decision boundary for a trained binary classifier.
    This method uses a contour plot and predictions from the input model
    to determine locations on a grid of points where predictions are 0 vs. 1.
    So this expectes a model that is trained on 2 features, and that is a
    binary classifier.
    
    This function is inspired/stolen from scikit-learn plot svm kernels example: 
    https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html
    
    Parameters
    ----------
    model - A scikit-learn trained classifier, when we call predict(X) on 
       the X features given as input, we expect binary predictions to be returned.
       NOTE: this funciton expects that the given trained model is actually a
       Pipeline.  It expects that the Pipeline has a 'scaler' step that is
       a standard scaler.  It expects that the Pipeline has a 'clf' step as the
       final step that is a support vector machine classifier.  This final
       `clf` should have a parameter indicating the support vectors that were
       fitt by the model.
    X     - The training data used to fit the model.  We expect X to contain 2
       features, so it should be an (m, 2) shaped array.
    y     - The true labels used to fit the model with.  Should be a vector of shape
        (m,) with the same number of lables as samples in the X training data
    remove_bias - For most models, there are only 2 features at indexes 0 and 1
        so the location of support vectors come from indexes (0, 1).  But when we
        perform a polynomial transformation by hand, we end up with a bias parameter/term
        at index 0, so need to use indexes 1 and 2 in that case from the support vectors
        reported.
        
    Returns
    -------
    No explicit result is returned.  But a matplotlib figure is created and a
    contour plot is drawn on this figure.
    '''
    # pick contrasting colors from Paired colormap for class 0 and class 1
    colors = plt.cm.Paired.colors
    c0, c1 = colors[0], colors[11]

    # plot the raw data using colors and markers for the two classes
    plt.plot(X[y == 0]['x_1'], X[y == 0]['x_2'], "o", color=c0, label=class_0_label, zorder=10, markeredgecolor='k')
    plt.plot(X[y == 1]['x_1'], X[y == 1]['x_2'], "s", color=c1, label=class_1_label, zorder=10, markeredgecolor='k')

    '''
    This section had to be removed becuase of errors when running the classifier with the custom kernel function.
    For some reason, the model['clf'] is empty in Task 3.3. I couldn't troubleshoot to find out why the previous classifier
    (using kernel='rbf') has the support_vectors_ available but the custom classifier does not.
    
    # indicate the support vectors used in the model classifier, we put a bigger 
    # circle around these to indcate the one used by svm classifier
    support_vectors = model['clf'].support_vectors_
    support_vectors_rescaled = model['scaler'].inverse_transform(support_vectors)
    x1_index, x2_index = 0,1
    if remove_bias:
        x1_index, x2_index = 1,2
    plt.scatter(support_vectors_rescaled[:, x1_index], support_vectors_rescaled[:, x2_index],
        s=150,
        facecolors="none",
        zorder=10,
        edgecolors="k",
    )
    '''

    # use determined figure axis limits to determine ranges for grid coverage
    plt.axis('tight')
    axes = plt.gca()
    x1_min, x1_max = axes.get_xlim()
    x2_min, x2_max = axes.get_ylim()

    # create grid of points over the plot to use contour plot to visualize decision boundary
    X1, X2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
    
    # use decision funciton of trained classifier to determine decisions over the grid
    Z = model.decision_function(np.c_[X1.ravel(), X2.ravel()])
    Z = Z.reshape(X1.shape)
    
    # visualize decision boundary, pcolormesh plots colors (c0, c1) on the mesh of points
    # where the decision was 0/1 respectively
    plt.pcolormesh(X1, X2, Z > 0, cmap=plt.cm.Paired)
    
    # visualize decision boundary and margins.  The decision function is before the sigmoid is done, thus where the level is 0
    # will be where the decision is 0.5, or in otherwords the decision boundary.
    # Levels of -1.0 and 1.0 correspond (roughly?) to the SVM RELU margin.  I.e. I believe this is roughly where the default
    # RELU transitions from horizontal to linear function
    CS = plt.contour(X1, X2, Z,
            colors=["k", "k", "k"],
            linestyles=["--", "-", "--"],
            levels=[-1.0, 0, 1.0],
         )

    # add labels and plot information
    plt.xlabel(feature_1_label, fontsize=14)
    plt.ylabel(feature_2_label, fontsize=14)
    
    # add legend, by hand get the contour line handles and add to handles and labels before we
    # display the legend
    h1, _ = CS.legend_elements()
    handles, labels = axes.get_legend_handles_labels()
    contour_labels = ['support margin', 'decision boundary', 'support margin']
    for h,l in zip(h1[1:], contour_labels[1:]):
        handles.append(h)
        labels.append(l)
    plt.legend(handles, labels, fontsize=14)