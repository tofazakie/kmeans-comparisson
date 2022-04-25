"""
The :mod:`sklearn.datasets` module includes utilities to load datasets,
including methods to load and fetch popular reference datasets. It also
features some artificial data generators.
"""
from .base import load_australian
from .base import load_audit
from .base import load_balance
from .base import load_banana
from .base import load_breast_cancer
from .base import load_boston
from .base import load_diabetes
from .base import load_diabetic
from .base import load_digits
from .base import load_files
from .base import load_ilpd
from .base import load_ionosphere
from .base import load_iris
from .base import load_banknote
from .base import load_bankmarket
from .base import load_banknote_sample
from .base import load_biodeg
from .base import load_blood
from .base import load_hepatitis
from .base import load_haberman
from .base import load_letter
from .base import load_spambase
from .base import load_wilt
from .base import load_breastcancer_winconsin
from .base import load_breast_tissue
from .base import load_ecoli
from .base import load_seeds
from .base import load_phising_data
from .base import load_glass
from .base import load_german
from .base import load_linnerud
from .base import load_mammographic
from .base import load_optdigits
from .base import load_pageblocks
from .base import load_penbased
from .base import load_pima_indian
from .base import load_sample_images
from .base import load_sample_image
from .base import load_sonar
from .base import load_segment
from .base import load_thyroid
from .base import load_vehicle
from .base import load_vowel
from .base import load_wine
from .base import load_yeast0vs4
from .base import get_data_home
from .base import clear_data_home
from .covtype import fetch_covtype
from .kddcup99 import fetch_kddcup99
from .mlcomp import load_mlcomp
from .lfw import fetch_lfw_pairs
from .lfw import fetch_lfw_people
from .twenty_newsgroups import fetch_20newsgroups
from .twenty_newsgroups import fetch_20newsgroups_vectorized
from .mldata import fetch_mldata, mldata_filename
from .samples_generator import make_classification
from .samples_generator import make_multilabel_classification
from .samples_generator import make_hastie_10_2
from .samples_generator import make_regression
from .samples_generator import make_blobs
from .samples_generator import make_moons
from .samples_generator import make_circles
from .samples_generator import make_friedman1
from .samples_generator import make_friedman2
from .samples_generator import make_friedman3
from .samples_generator import make_low_rank_matrix
from .samples_generator import make_sparse_coded_signal
from .samples_generator import make_sparse_uncorrelated
from .samples_generator import make_spd_matrix
from .samples_generator import make_swiss_roll
from .samples_generator import make_s_curve
from .samples_generator import make_sparse_spd_matrix
from .samples_generator import make_gaussian_quantiles
from .samples_generator import make_biclusters
from .samples_generator import make_checkerboard
from .svmlight_format import load_svmlight_file
from .svmlight_format import load_svmlight_files
from .svmlight_format import dump_svmlight_file
from .olivetti_faces import fetch_olivetti_faces
from .species_distributions import fetch_species_distributions
from .california_housing import fetch_california_housing
from .rcv1 import fetch_rcv1


__all__ = ['clear_data_home',
           'dump_svmlight_file',
           'fetch_20newsgroups',
           'fetch_20newsgroups_vectorized',
           'fetch_lfw_pairs',
           'fetch_lfw_people',
           'fetch_mldata',
           'fetch_olivetti_faces',
           'fetch_species_distributions',
           'fetch_california_housing',
           'fetch_covtype',
           'fetch_rcv1',
           'fetch_kddcup99',
           'get_data_home',
           'load_australian',
           'load_audit',
           'load_balance',
           'load_banana',
           'load_boston',
           'load_diabetes',
           'load_diabetic',
           'load_digits',
           'load_files',
           'load_hepatitis',
           'load_haberman',
           'load_ilpd',
           'load_ionosphere',
           'load_iris',
           'load_banknote',
           'load_bankmarket',
           'load_banknote_sample',
           'load_biodeg',
           'load_blood',
           'load_spambase',
           'load_wilt',
           'load_breastcancer_winconsin',
           'load_breast_cancer',
           'load_breast_tissue',
           'load_ecoli',
           'load_seeds',
           'load_phising_data',
           'load_glass',
           'load_german',
           'load_letter',
           'load_linnerud',
           'load_mammographic',
           'load_optdigits',
           'load_pageblocks',
           'load_penbased',
           'load_segment',
           'load_sonar',
           'load_yeast0vs4',
           'load_mlcomp',
           'load_sample_image',
           'load_sample_images',
           'load_svmlight_file',
           'load_svmlight_files',
           'load_thyroid',
           'load_pima_indian',
           'load_vehicle',
           'load_vowel',
           'load_wine',
           'make_biclusters',
           'make_blobs',
           'make_circles',
           'make_classification',
           'make_checkerboard',
           'make_friedman1',
           'make_friedman2',
           'make_friedman3',
           'make_gaussian_quantiles',
           'make_hastie_10_2',
           'make_low_rank_matrix',
           'make_moons',
           'make_multilabel_classification',
           'make_regression',
           'make_s_curve',
           'make_sparse_coded_signal',
           'make_sparse_spd_matrix',
           'make_sparse_uncorrelated',
           'make_spd_matrix',
           'make_swiss_roll',
           'mldata_filename']


