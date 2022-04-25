"""
Base IO code for all datasets
"""

# Copyright (c) 2007 David Cournapeau <cournape@gmail.com>
#               2010 Fabian Pedregosa <fabian.pedregosa@inria.fr>
#               2010 Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause
from __future__ import print_function

import os
import csv
import sys
import shutil
from collections import namedtuple
from os import environ, listdir, makedirs
from os.path import dirname, exists, expanduser, isdir, join, splitext
import hashlib

from ..utils import Bunch
from ..utils import check_random_state

import numpy as np

from sklearn.externals.six.moves.urllib.request import urlretrieve

RemoteFileMetadata = namedtuple('RemoteFileMetadata',
                                ['filename', 'url', 'checksum'])


def get_data_home(data_home=None):
    """Return the path of the scikit-learn data dir.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times.

    By default the data dir is set to a folder named 'scikit_learn_data' in the
    user home folder.

    Alternatively, it can be set by the 'SCIKIT_LEARN_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home : str | None
        The path to scikit-learn data dir.
    """
    if data_home is None:
        data_home = environ.get('SCIKIT_LEARN_DATA',
                                join('~', 'scikit_learn_data'))
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home


def clear_data_home(data_home=None):
    """Delete all the content of the data home cache.

    Parameters
    ----------
    data_home : str | None
        The path to scikit-learn data dir.
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def load_files(container_path, description=None, categories=None,
               load_content=True, shuffle=True, encoding=None,
               decode_error='strict', random_state=0):
    """Load text files with categories as subfolder names.

    Individual samples are assumed to be files stored a two levels folder
    structure such as the following:

        container_folder/
            category_1_folder/
                file_1.txt
                file_2.txt
                ...
                file_42.txt
            category_2_folder/
                file_43.txt
                file_44.txt
                ...

    The folder names are used as supervised signal label names. The individual
    file names are not important.

    This function does not try to extract features into a numpy array or scipy
    sparse matrix. In addition, if load_content is false it does not try to
    load the files in memory.

    To use text files in a scikit-learn classification or clustering algorithm,
    you will need to use the `sklearn.feature_extraction.text` module to build
    a feature extraction transformer that suits your problem.

    If you set load_content=True, you should also specify the encoding of the
    text using the 'encoding' parameter. For many modern text files, 'utf-8'
    will be the correct encoding. If you leave encoding equal to None, then the
    content will be made of bytes instead of Unicode, and you will not be able
    to use most functions in `sklearn.feature_extraction.text`.

    Similar feature extractors should be built for other kind of unstructured
    data input such as images, audio, video, ...

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category

    description : string or unicode, optional (default=None)
        A paragraph describing the characteristic of the dataset: its source,
        reference, etc.

    categories : A collection of strings or None, optional (default=None)
        If None (default), load all the categories. If not None, list of
        category names to load (other categories ignored).

    load_content : boolean, optional (default=True)
        Whether to load or not the content of the different files. If true a
        'data' attribute containing the text information is present in the data
        structure returned. If not, a filenames attribute gives the path to the
        files.

    shuffle : bool, optional (default=True)
        Whether or not to shuffle the data: might be important for models that
        make the assumption that the samples are independent and identically
        distributed (i.i.d.), such as stochastic gradient descent.

    encoding : string or None (default is None)
        If None, do not try to decode the content of the files (e.g. for images
        or other non-text content). If not None, encoding to use to decode text
        files to Unicode if load_content is True.

    decode_error : {'strict', 'ignore', 'replace'}, optional
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. Passed as keyword
        argument 'errors' to bytes.decode.

    random_state : int, RandomState instance or None, optional (default=0)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are: either
        data, the raw text data to learn, or 'filenames', the files
        holding it, 'target', the classification labels (integer index),
        'target_names', the meaning of the labels, and 'DESCR', the full
        description of the dataset.
    """
    target = []
    target_names = []
    filenames = []

    folders = [f for f in sorted(listdir(container_path))
               if isdir(join(container_path, f))]

    if categories is not None:
        folders = [f for f in folders if f in categories]

    for label, folder in enumerate(folders):
        target_names.append(folder)
        folder_path = join(container_path, folder)
        documents = [join(folder_path, d)
                     for d in sorted(listdir(folder_path))]
        target.extend(len(documents) * [label])
        filenames.extend(documents)

    # convert to array for fancy indexing
    filenames = np.array(filenames)
    target = np.array(target)

    if shuffle:
        random_state = check_random_state(random_state)
        indices = np.arange(filenames.shape[0])
        random_state.shuffle(indices)
        filenames = filenames[indices]
        target = target[indices]

    if load_content:
        data = []
        for filename in filenames:
            with open(filename, 'rb') as f:
                data.append(f.read())
        if encoding is not None:
            data = [d.decode(encoding, decode_error) for d in data]
        return Bunch(data=data,
                     filenames=filenames,
                     target_names=target_names,
                     target=target,
                     DESCR=description)

    return Bunch(filenames=filenames,
                 target_names=target_names,
                 target=target,
                 DESCR=description)


def load_data(module_path, data_file_name):
    """Loads data from module_path/data/data_file_name.

    Parameters
    ----------
    data_file_name : String. Name of csv file to be loaded from
    module_path/data/data_file_name. For example 'wine_data.csv'.

    Returns
    -------
    data : Numpy Array
        A 2D array with each row representing one sample and each column
        representing the features of a given sample.

    target : Numpy Array
        A 1D array holding target variables for all the samples in `data.
        For example target[0] is the target varible for data[0].

    target_names : Numpy Array
        A 1D array containing the names of the classifications. For example
        target_names[0] is the name of the target[0] class.
    """
    with open(join(module_path, 'data', data_file_name)) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=np.int)

    return data, target, target_names


def load_wine(return_X_y=False):
    """Load and return the wine dataset (classification).

    .. versionadded:: 0.18

    The wine dataset is a classic and very easy multi-class classification
    dataset.

    =================   ==============
    Classes                          3
    Samples per class        [59,71,48]
    Samples total                  178
    Dimensionality                  13
    Features            real, positive
    =================   ==============

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are: 'data', the
        data to learn, 'target', the classification labels, 'target_names', the
        meaning of the labels, 'feature_names', the meaning of the features,
        and 'DESCR', the full description of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

    The copy of UCI ML Wine Data Set dataset is downloaded and modified to fit
    standard format from:
    https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data

    Examples
    --------
    Let's say you are interested in the samples 10, 80, and 140, and want to
    know their class name.

    >>> from sklearn.datasets import load_wine
    >>> data = load_wine()
    >>> data.target[[10, 80, 140]]
    array([0, 1, 2])
    >>> list(data.target_names)
    ['class_0', 'class_1', 'class_2']
    """
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'wine_data.csv')

    with open(join(module_path, 'descr', 'wine_data.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['alcohol',
                                'malic_acid',
                                'ash',
                                'alcalinity_of_ash',
                                'magnesium',
                                'total_phenols',
                                'flavanoids',
                                'nonflavanoid_phenols',
                                'proanthocyanins',
                                'color_intensity',
                                'hue',
                                'od280/od315_of_diluted_wines',
                                'proline'])


def load_iris(return_X_y=False):
    """Load and return the iris dataset (classification).

    The iris dataset is a classic and very easy multi-class classification
    dataset.

    =================   ==============
    Classes                          3
    Samples per class               50
    Samples total                  150
    Dimensionality                   4
    Features            real, positive
    =================   ==============

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object. See
        below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification labels,
        'target_names', the meaning of the labels, 'feature_names', the
        meaning of the features, and 'DESCR', the
        full description of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.18

    Examples
    --------
    Let's say you are interested in the samples 10, 25, and 50, and want to
    know their class name.

    >>> from sklearn.datasets import load_iris
    >>> data = load_iris()
    >>> data.target[[10, 25, 50]]
    array([0, 0, 1])
    >>> list(data.target_names)
    ['setosa', 'versicolor', 'virginica']
    """
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'iris.csv')

    with open(join(module_path, 'descr', 'iris.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['sepal length (cm)', 'sepal width (cm)',
                                'petal length (cm)', 'petal width (cm)'])


def load_breast_cancer(return_X_y=False):
    """Load and return the breast cancer wisconsin dataset (classification).

    The breast cancer dataset is a classic and very easy binary classification
    dataset.

    =================   ==============
    Classes                          2
    Samples per class    212(M),357(B)
    Samples total                  569
    Dimensionality                  30
    Features            real, positive
    =================   ==============

    Parameters
    ----------
    return_X_y : boolean, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification labels,
        'target_names', the meaning of the labels, 'feature_names', the
        meaning of the features, and 'DESCR', the
        full description of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.18

    The copy of UCI ML Breast Cancer Wisconsin (Diagnostic) dataset is
    downloaded from:
    https://goo.gl/U2Uwz2

    Examples
    --------
    Let's say you are interested in the samples 10, 50, and 85, and want to
    know their class name.

    >>> from sklearn.datasets import load_breast_cancer
    >>> data = load_breast_cancer()
    >>> data.target[[10, 50, 85]]
    array([0, 1, 0])
    >>> list(data.target_names)
    ['malignant', 'benign']
    """
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'breast_cancer.csv')

    with open(join(module_path, 'descr', 'breast_cancer.rst')) as rst_file:
        fdescr = rst_file.read()

    feature_names = np.array(['mean radius', 'mean texture',
                              'mean perimeter', 'mean area',
                              'mean smoothness', 'mean compactness',
                              'mean concavity', 'mean concave points',
                              'mean symmetry', 'mean fractal dimension',
                              'radius error', 'texture error',
                              'perimeter error', 'area error',
                              'smoothness error', 'compactness error',
                              'concavity error', 'concave points error',
                              'symmetry error', 'fractal dimension error',
                              'worst radius', 'worst texture',
                              'worst perimeter', 'worst area',
                              'worst smoothness', 'worst compactness',
                              'worst concavity', 'worst concave points',
                              'worst symmetry', 'worst fractal dimension'])

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=feature_names)


def load_digits(n_class=10, return_X_y=False):
    """Load and return the digits dataset (classification).

    Each datapoint is a 8x8 image of a digit.

    =================   ==============
    Classes                         10
    Samples per class             ~180
    Samples total                 1797
    Dimensionality                  64
    Features             integers 0-16
    =================   ==============

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    n_class : integer, between 0 and 10, optional (default=10)
        The number of classes to return.

    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'images', the images corresponding
        to each sample, 'target', the classification labels for each
        sample, 'target_names', the meaning of the labels, and 'DESCR',
        the full description of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.18

    This is a copy of the test set of the UCI ML hand-written digits datasets
    http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

    Examples
    --------
    To load the data and visualize the images::

        >>> from sklearn.datasets import load_digits
        >>> digits = load_digits()
        >>> print(digits.data.shape)
        (1797, 64)
        >>> import matplotlib.pyplot as plt #doctest: +SKIP
        >>> plt.gray() #doctest: +SKIP
        >>> plt.matshow(digits.images[0]) #doctest: +SKIP
        >>> plt.show() #doctest: +SKIP
    """
    module_path = dirname(__file__)
    # data = np.loadtxt(join(module_path, 'data', 'digits.csv'),
    #                   delimiter=',')
    # with open(join(module_path, 'descr', 'digits.rst')) as f:
    #     descr = f.read()
    # target = data[:, -1].astype(np.int)
    # flat_data = data[:, :-1]
    # images = flat_data.view()
    # images.shape = (-1, 8, 8)

    # if n_class < 10:
    #     idx = target < n_class
    #     flat_data, target = flat_data[idx], target[idx]
    #     images = images[idx]

    # if return_X_y:
    #     return flat_data, target

    # return Bunch(data=flat_data,
    #              target=target,
    #              target_names=np.arange(10),
    #              images=images,
    #              DESCR=descr)


    data, target, target_names = load_data(module_path, 'digits.csv')

    with open(join(module_path, 'descr', 'digits.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['att1', 'att2', 'att3', 'att4', 'att5', 'att6', 'att7', 'att8', 'att9', 'att10',
                                'att11', 'att12', 'att13', 'att14', 'att15', 'att16', 'att17', 'att18', 'att19', 'att20',
                                'att21', 'att22', 'att23', 'att24', 'att25', 'att26', 'att27', 'att28', 'att29', 'att30',
                                'att31', 'att32', 'att33', 'att34', 'att35', 'att36', 'att37', 'att38', 'att39', 'att40',
                                'att41', 'att42', 'att43', 'att44', 'att45', 'att46', 'att47', 'att48', 'att49', 'att50',
                                'att51', 'att52', 'att53', 'att54', 'att55', 'att56', 'att57', 'att58', 'att59', 'att60',
                                'att61', 'att62', 'att63', 'att64', 'class']
                 )


def load_optdigits(n_class=10, return_X_y=False):
    
    module_path = dirname(__file__)
    
    data, target, target_names = load_data(module_path, 'optdigits.csv')

    with open(join(module_path, 'descr', 'optdigits.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['att1', 'att2', 'att3', 'att4', 'att5', 'att6', 'att7', 'att8', 'att9', 'att10',
                                'att11', 'att12', 'att13', 'att14', 'att15', 'att16', 'att17', 'att18', 'att19', 'att20',
                                'att21', 'att22', 'att23', 'att24', 'att25', 'att26', 'att27', 'att28', 'att29', 'att30',
                                'att31', 'att32', 'att33', 'att34', 'att35', 'att36', 'att37', 'att38', 'att39', 'att40',
                                'att41', 'att42', 'att43', 'att44', 'att45', 'att46', 'att47', 'att48', 'att49', 'att50',
                                'att51', 'att52', 'att53', 'att54', 'att55', 'att56', 'att57', 'att58', 'att59', 'att60',
                                'att61', 'att62', 'att63', 'att64', 'class']
                 )



def load_diabetes(return_X_y=False):
    """Load and return the diabetes dataset (regression).

    ==============      ==================
    Samples total       442
    Dimensionality      10
    Features            real, -.2 < x < .2
    Targets             integer 25 - 346
    ==============      ==================

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn and 'target', the regression target for each
        sample.

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.18
    """

    module_path = dirname(__file__)
    base_dir = join(module_path, 'data')
    data = np.loadtxt(join(base_dir, 'diabetes_data.csv.gz'))
    target = np.loadtxt(join(base_dir, 'diabetes_target.csv.gz'))

    with open(join(module_path, 'descr', 'diabetes.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target, DESCR=fdescr,
                 feature_names=['age', 'sex', 'bmi', 'bp',
                                's1', 's2', 's3', 's4', 's5', 's6'])


def load_linnerud(return_X_y=False):
    """Load and return the linnerud dataset (multivariate regression).

    ==============    ============================
    Samples total     20
    Dimensionality    3 (for both data and target)
    Features          integer
    Targets           integer
    ==============    ============================

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are: 'data' and
        'targets', the two multivariate datasets, with 'data' corresponding to
        the exercise and 'targets' corresponding to the physiological
        measurements, as well as 'feature_names' and 'target_names'.

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.18
    """
    base_dir = join(dirname(__file__), 'data/')
    # Read data
    data_exercise = np.loadtxt(base_dir + 'linnerud_exercise.csv', skiprows=1)
    data_physiological = np.loadtxt(base_dir + 'linnerud_physiological.csv',
                                    skiprows=1)
    # Read header
    with open(base_dir + 'linnerud_exercise.csv') as f:
        header_exercise = f.readline().split()
    with open(base_dir + 'linnerud_physiological.csv') as f:
        header_physiological = f.readline().split()
    with open(dirname(__file__) + '/descr/linnerud.rst') as f:
        descr = f.read()

    if return_X_y:
        return data_exercise, data_physiological

    return Bunch(data=data_exercise, feature_names=header_exercise,
                 target=data_physiological,
                 target_names=header_physiological,
                 DESCR=descr)


def load_boston(return_X_y=False):
    """Load and return the boston house-prices dataset (regression).

    ==============     ==============
    Samples total                 506
    Dimensionality                 13
    Features           real, positive
    Targets             real 5. - 50.
    ==============     ==============

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the regression targets,
        and 'DESCR', the full description of the dataset.

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.18

    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> boston = load_boston()
    >>> print(boston.data.shape)
    (506, 13)
    """
    module_path = dirname(__file__)

    fdescr_name = join(module_path, 'descr', 'boston_house_prices.rst')
    with open(fdescr_name) as f:
        descr_text = f.read()

    data_file_name = join(module_path, 'data', 'boston_house_prices.csv')
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,))
        temp = next(data_file)  # names of features
        feature_names = np.array(temp)

        for i, d in enumerate(data_file):
            data[i] = np.asarray(d[:-1], dtype=np.float64)
            target[i] = np.asarray(d[-1], dtype=np.float64)

    if return_X_y:
        return data, target

    return Bunch(data=data,
                 target=target,
                 # last column is target value
                 feature_names=feature_names[:-1],
                 DESCR=descr_text)


def load_sample_images():
    """Load sample images for image manipulation.

    Loads both, ``china`` and ``flower``.

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes : 'images', the
        two sample images, 'filenames', the file names for the images, and
        'DESCR' the full description of the dataset.

    Examples
    --------
    To load the data and visualize the images:

    >>> from sklearn.datasets import load_sample_images
    >>> dataset = load_sample_images()     #doctest: +SKIP
    >>> len(dataset.images)                #doctest: +SKIP
    2
    >>> first_img_data = dataset.images[0] #doctest: +SKIP
    >>> first_img_data.shape               #doctest: +SKIP
    (427, 640, 3)
    >>> first_img_data.dtype               #doctest: +SKIP
    dtype('uint8')
    """
    # Try to import imread from scipy. We do this lazily here to prevent
    # this module from depending on PIL.
    try:
        try:
            from scipy.misc import imread
        except ImportError:
            from scipy.misc.pilutil import imread
    except ImportError:
        raise ImportError("The Python Imaging Library (PIL) "
                          "is required to load data from jpeg files")
    module_path = join(dirname(__file__), "images")
    with open(join(module_path, 'README.txt')) as f:
        descr = f.read()
    filenames = [join(module_path, filename)
                 for filename in os.listdir(module_path)
                 if filename.endswith(".jpg")]
    # Load image data for each image in the source folder.
    images = [imread(filename) for filename in filenames]

    return Bunch(images=images,
                 filenames=filenames,
                 DESCR=descr)


def load_sample_image(image_name):
    """Load the numpy array of a single sample image

    Parameters
    -----------
    image_name : {`china.jpg`, `flower.jpg`}
        The name of the sample image loaded

    Returns
    -------
    img : 3D array
        The image as a numpy array: height x width x color

    Examples
    ---------

    >>> from sklearn.datasets import load_sample_image
    >>> china = load_sample_image('china.jpg')   # doctest: +SKIP
    >>> china.dtype                              # doctest: +SKIP
    dtype('uint8')
    >>> china.shape                              # doctest: +SKIP
    (427, 640, 3)
    >>> flower = load_sample_image('flower.jpg') # doctest: +SKIP
    >>> flower.dtype                             # doctest: +SKIP
    dtype('uint8')
    >>> flower.shape                             # doctest: +SKIP
    (427, 640, 3)
    """
    images = load_sample_images()
    index = None
    for i, filename in enumerate(images.filenames):
        if filename.endswith(image_name):
            index = i
            break
    if index is None:
        raise AttributeError("Cannot find sample image: %s" % image_name)
    return images.images[index]


def _pkl_filepath(*args, **kwargs):
    """Ensure different filenames for Python 2 and Python 3 pickles

    An object pickled under Python 3 cannot be loaded under Python 2. An object
    pickled under Python 2 can sometimes not be loaded correctly under Python 3
    because some Python 2 strings are decoded as Python 3 strings which can be
    problematic for objects that use Python 2 strings as byte buffers for
    numerical data instead of "real" strings.

    Therefore, dataset loaders in scikit-learn use different files for pickles
    manages by Python 2 and Python 3 in the same SCIKIT_LEARN_DATA folder so as
    to avoid conflicts.

    args[-1] is expected to be the ".pkl" filename. Under Python 3, a suffix is
    inserted before the extension to s

    _pkl_filepath('/path/to/folder', 'filename.pkl') returns:
      - /path/to/folder/filename.pkl under Python 2
      - /path/to/folder/filename_py3.pkl under Python 3+

    """
    py3_suffix = kwargs.get("py3_suffix", "_py3")
    basename, ext = splitext(args[-1])
    if sys.version_info[0] >= 3:
        basename += py3_suffix
    new_args = args[:-1] + (basename + ext,)
    return join(*new_args)


def _sha256(path):
    """Calculate the sha256 hash of the file at path."""
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(path, "rb") as f:
        while True:
            buffer = f.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)
    return sha256hash.hexdigest()


def _fetch_remote(remote, dirname=None):
    """Helper function to download a remote dataset into path

    Fetch a dataset pointed by remote's url, save into path using remote's
    filename and ensure its integrity based on the SHA256 Checksum of the
    downloaded file.

    Parameters
    -----------
    remote : RemoteFileMetadata
        Named tuple containing remote dataset meta information: url, filename
        and checksum

    dirname : string
        Directory to save the file to.

    Returns
    -------
    file_path: string
        Full path of the created file.
    """

    file_path = (remote.filename if dirname is None
                 else join(dirname, remote.filename))
    urlretrieve(remote.url, file_path)
    checksum = _sha256(file_path)
    if remote.checksum != checksum:
        raise IOError("{} has an SHA256 checksum ({}) "
                      "differing from expected ({}), "
                      "file may be corrupted.".format(file_path, checksum,
                                                      remote.checksum))
    return file_path



def load_banknote(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'banknote.csv')

    with open(join(module_path, 'descr', 'banknote.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['variance', 'skewness',
                                'curtosis', 'entropy'])


def load_banknote_sample(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'banknote_sample.csv')

    with open(join(module_path, 'descr', 'banknote_sample.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['variance', 'skewness',
                                'curtosis', 'entropy'])


def load_biodeg(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'biodeg.csv')

    with open(join(module_path, 'descr', 'biodeg.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['SpMax_L','J_Dz(e)','nHM','F01[N-N]','F04[C-N]',
                                'NssssC','nCb-','C%','nCp','nO',
                                'F03[C-N]','SdssC','HyWi_B(m)','LOC','SM6_L',
                                'F03[C-O]','Me','Mi','nN-N','nArNO2',
                                'nCRX3','SpPosA_B(p)','nCIR','B01[C-Br]','B03[C-Cl]',
                                'N-073','SpMax_A','Psi_i_1d','B04[C-Br]','SdO,TI2_L',
                                'nCrt','C-026','F02[C-N]','nHDon','SpMax_B',
                                'Psi_i_A','nN','SM6_B(m)','nArCOOR','nX'])


def load_spambase(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'spambase.csv')

    with open(join(module_path, 'descr', 'spambase.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['word_freq_make','word_freq_address','word_freq_all','word_freq_3d','word_freq_our',
                                'word_freq_over','word_freq_remove','wordfreq_internet','word_freq_order','word_freq_mail',
                                'word_freq_receive','word_freq_will','word_freq_people','word_freq_repor','word_freq_addresses',
                                'word_freq_free','word_freq_business','word_freq_email','word_freq_you','word_freq_credit',
                                'word_freq_your','word_freq_font','word_freq_000','word_freq_money','word_freq_hp',
                                'word_freq_hpl','word_freq_george','word_freq_650','word_freq_lab','word_freq_labs',
                                'word_freq_telnet','word_freq_857','word_freq_data','word_freq_415','word_freq_85',
                                'word_freq_technology','word_freq_1999','word_freq_parts','word_freq_pm','word_freq_direct',
                                'word_freq_cs','word_freq_meeting','word_freq_original','word_freq_project','word_freq_re',
                                'word_freq_edu','word_freq_table','word_freq_conference','char_freq_;','char_freq_(',
                                'char_freq_[','char_freq_!','char_freq_$','char_freq_#','capital_run_length_average',
                                'capital_run_length_longest','capital_run_length_total'])


def load_wilt(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'wilt.csv')

    with open(join(module_path, 'descr', 'wilt.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['GLCM_pan', 'Mean_Green', 'Mean_Red',
                                'Mean_NIR', 'SD_pan'])


def load_bankmarket(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'bankmarket.csv')

    with open(join(module_path, 'descr', 'bankmarket.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['age', 'job', 'education','default','balance','housing','loan'
                                'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y'])




def load_breastcancer_winconsin(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'breastcancer_winconsin.csv')

    with open(join(module_path, 'descr', 'breastcancer_winconsin.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['clump_thickness', 'uniformity_of_cell_size ', 'uniformity_of_cell_shape','marginal_adhesion','single_epithelial_cell_size',
                                'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class'])




def load_breast_tissue(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'breast_tissue.csv')

    with open(join(module_path, 'descr', 'breast_tissue.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['I0', 'PA500 ', 'HFS','DA','Area',
                                'A/DA', 'Max IP', 'DR', 'P', 'class'])



def load_ecoli(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'ecoli.csv')

    with open(join(module_path, 'descr', 'ecoli.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['mcg', 'gvh ', 'lip','chg','aac',
                                'alm1', 'alm2', 'class'])



def load_seeds(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'seeds.csv')

    with open(join(module_path, 'descr', 'seeds.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['area_a', 'perimeter_p', 'compactness','kernel_length','kernel_width',
                                'asymmetry_coef', 'kernel_groove_length', 'class'])


def load_phising_data(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'phising_data.csv')

    with open(join(module_path, 'descr', 'phising_data.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['area_a', 'perimeter_p', 'compactness','kernel_length','kernel_width',
                                'asymmetry_coef', 'kernel_groove_length', 'class'])



def load_glass(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'glass.csv')

    with open(join(module_path, 'descr', 'glass.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['RI', 'Na', 'Mg','Al','Si','K',
                                'Ca', 'Ba', 'Fe', 'class'])

def load_australian(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'australian.csv')

    with open(join(module_path, 'descr', 'australian.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7',
                                'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'class'])


def load_blood(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'blood.csv')

    with open(join(module_path, 'descr', 'blood.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['R', 'F', 'M', 'T', 'class'])



def load_audit(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'blood.csv')

    with open(join(module_path, 'descr', 'blood.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['Sector_score', 'LOCATION_ID', 'PARA_A', 'SCORE_A', 'PARA_B', 'SCORE_B',
                 'TOTAL', 'numbers', 'Marks', 'Money_Value', 'MONEY_Marks', 'District', 'Loss', 'LOSS_SCORE',
                 'History', 'History_score', 'Score', 'class'])


def load_diabetic(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'diabetic.csv')

    with open(join(module_path, 'descr', 'diabetic.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['qa', 'pre-screen', 'MA1', 'MA2', 'MA3', 'MA4', 'MA5', 'MA6', 
                                'ex1', 'ex2', 'ex3', 'ex4', 'ex5', 'ex6', 
                                'dist', 'diameter', '', 'class'])


def load_mammographic(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'mammographic.csv')

    with open(join(module_path, 'descr', 'mammographic.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['BI-RADS', 'age', 'shape', 'margin', 'density', 'class'])


def load_ilpd(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'ilpd.csv')

    with open(join(module_path, 'descr', 'ilpd.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['age', 'gender', 'tb', 'db', 'AP', 'Sgpt-AA', 
                                'Sgot-AA', 'TP', 'ALB', 'A/G', 'class'])


def load_yeast0vs4(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'yeast0vs4.csv')

    with open(join(module_path, 'descr', 'yeast0vs4.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 
                                'vac', 'nuc', 'class'])


def load_german(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'german.csv')

    with open(join(module_path, 'descr', 'german.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 
                                'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20',
                                'a21', 'a22', 'a23', 'a24', 'class'])



def load_pima_indian(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'pima_indian.csv')

    with open(join(module_path, 'descr', 'pima_indian.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['N pregnant', 'glucose concentration', 'diastolic', 'triceps', 'serum', 'BMI', 
                 'diabet pedigree', 'age', 'class'])


def load_balance(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'balance.csv')

    with open(join(module_path, 'descr', 'balance.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance', 'class'])


def load_banana(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'banana.csv')

    with open(join(module_path, 'descr', 'banana.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['Attr-1', 'Attr-2', 'class'])


def load_hepatitis(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'hepatitis.csv')

    with open(join(module_path, 'descr', 'hepatitis.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['age', 'sex', 'steroid', 'antivirals', 'fatigue', 'malaise', 'anorexia',
                 'liver_big', 'liver_firm', 'spleen', 'spiders', 'ascites', 'varices', 'bilirubin', 'alk', 
                 'sgot', 'albumin', 'protime', 'histology', 'class'])


def load_ionosphere(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'ionosphere.csv')

    with open(join(module_path, 'descr', 'ionosphere.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['par1', 'par2', 'par3', 'par4', 'par5', 'par6', 'par7', 'par8', 'par9', 'par10',
                 'par11', 'par12', 'par13', 'par14', 'par15', 'par16', 'par17', 'par18', 'par19', 'par20',
                 'par21', 'par22', 'par23', 'par24', 'par25', 'par26', 'par27', 'par28', 'par29', 'par30',
                 'par31', 'par32', 'par33', 'par34', 'class'])


def load_sonar(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'sonar.csv')

    with open(join(module_path, 'descr', 'sonar.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['par1', 'par2', 'par3', 'par4', 'par5', 'par6', 'par7', 'par8', 'par9', 'par10',
                 'par11', 'par12', 'par13', 'par14', 'par15', 'par16', 'par17', 'par18', 'par19', 'par20',
                 'par21', 'par22', 'par23', 'par24', 'par25', 'par26', 'par27', 'par28', 'par29', 'par30',
                 'par31', 'par32', 'par33', 'par34', 'par35', 'par36', 'par37', 'par38', 'par39', 'par40',
                 'par41', 'par42', 'par43', 'par44', 'par45', 'par46', 'par47', 'par48', 'par49', 'par50',
                 'par51', 'par52', 'par53', 'par54', 'par55', 'par56', 'par57', 'par58', 'par59', 'par60',
                 'class'])


def load_letter(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'letter.csv')

    with open(join(module_path, 'descr', 'letter.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar', 'y-bar', 
                 'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx', 'class'])


def load_thyroid(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'thyroid.csv')

    with open(join(module_path, 'descr', 'thyroid.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['T3-resin', 'Serum thyroxin', 'triiodothyronine', 'TSH', 'Maximal absolute', 'class'])


def load_segment(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'segment.csv')

    with open(join(module_path, 'descr', 'segment.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['region-centroid-col', 'region-centroid-row', 'region-pixel-count', 'short-line-density-5', 'short-line-density-2', 
                 'vedge-mean', 'vegde-sd', 'hedge-mean', 'hedge-sd', 'intensity-mean', 'rawred-mean', 'rawblue-mean', 'rawgreen-mean',
                 'exred-mean', 'exblue-mean', 'exgreen-mean', 'value-mean', 'saturatoin-mean', 'hue-mean', 'class'])



def load_vehicle(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'vehicle.csv')

    with open(join(module_path, 'descr', 'vehicle.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['COMPACTNESS', 'CIRCULARITY', 'DISTANCE-CIRCULARITY', 'RADIUS-RATIO', 'PR.AXIS-ASPECT-RATIO', 
                 'MAX.LENGTH-ASPECT-RATIO', 'SCATTER-RATIO', 'ELONGATEDNESS', 'PR.AXIS-RECTANGULARITY', 'MAX.LENGTH-RECTANGULARITY', 
                 'SCALED-VARIANCE-MAJOR-AXIS', 'SCALED-VARIANCE-MINOR-AXIS', 'SCALED-RADIUS-OF-GYRATION',
                 'SKEWNESS-ABOUT-MAJOR-AXIS', 'SKEWNESS-ABOUT-MINOR-AXIS', 'KURTOSIS-ABOUT-MAJOR-AXIS', 
                 'KURTOSIS-ABOUT-MINOR-AXIS', 'HOLLOWS-RATIO', 'class'])


def load_haberman(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'haberman.csv')

    with open(join(module_path, 'descr', 'haberman.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['age', 'year', 'positive', 'class'])


def load_pageblocks(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'page-blocks.csv')

    with open(join(module_path, 'descr', 'page-blocks.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['Height', 'Lenght', 'Area', 'Eccen', 'P_black', 'P_and', 'Mean_tr', 
                                'Blackpix', 'Blackand', 'Wb_trans', 'class'])


def load_vowel(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'vowel.csv')

    with open(join(module_path, 'descr', 'vowel.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['TT', 'SpeakerNumber', 'Sex', 'F0', 'F1', 'F2', 'F3', 'F4', 'F5', 
                                'F6', 'F7', 'F8', 'F9', 'class'])



def load_penbased(return_X_y=False):
    
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path, 'penbased.csv')

    with open(join(module_path, 'descr', 'penbased.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['At1', 'At2', 'At3', 'At4', 'At5', 'At6', 'At7', 'At8', 'At9', 'At10',
                 'At11', 'At12', 'At13', 'At14', 'At15', 'At16', 'class'])