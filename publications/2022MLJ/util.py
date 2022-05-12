from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier

from sklearn.decomposition import FastICA, KernelPCA, PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, Normalizer, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler
from sklearn.feature_selection import SelectPercentile, GenericUnivariateSelect, VarianceThreshold


def recover_pipeline(pl_as_str):
    if type(pl_as_str) != str:
        raise Exception()
    if pl_as_str[:8] != "Pipeline":
        raise Exception()
    return eval(pl_as_str)