import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis, NMF
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class DimensionalityReduction:
    """
    Dimensionality Reduction.
    This class can be used for dimensionality reduction and plotting of the result.

    Parameters:
        method (str): Dimensionality reduction method supports: `pca`, `svd`, `lda`, `t_sne`, `isomap`, `lle`, `fa`,
            `nmf`.

    Attributes:
        method (str): Dimensionality reduction method.
        estimator: Created model.
        X_transformed (pandas.DataFrame): Transformed X.
        methods_dict (dict): Methods dictionary.

    """

    def __init__(self, method='pca'):
        self.method = method
        self.estimator, self.X_transformed = None, None

    def transform(self, X, y=None, **kwargs):
        """
        Main dimensionality reduction method. It creates an estimator and fit_transform() given values.

        Parameters:
            X: X-value.
            y: y-value.
            **kwargs: Arguments for the estimator.

        Returns:
            X_transformed (pandas.DataFrame): New X dataframe with transformed values.

        Raises:
            NotImplementedError: If method is not supported.
        """
        # initialize all methods
        self._init_methods()

        # raise error if the method is not supported
        if self.method not in self.methods_dict.keys():
            raise NotImplementedError(f'Method {self.method} is not supported.')

        # get estimator and create transformed DataFrame
        self.estimator = self.methods_dict[self.method]
        self.X_transformed = pd.DataFrame(self.estimator(**kwargs).fit_transform(X=X, y=y))

        return self.X_transformed

    def plot_transformed(self, y, figsize=(10, 10), **kwargs):
        """
        Plot transformed X values using y as hue.
        If n_components < 3 it will use seaborn.scatterplot to plot values.
        Else it will use sns.pairplot to create plots.

        Note:
            This method can be called only after method 'transform' has
            been called.

        Parameters:
            y (pandas.Series, pandas.DataFrame): y-value.
            figsize (list), default=(10,10): Figure size.
            **kwargs: Arguments for the plot function.

        Raises:
            TypeError: If y is not pandas.DataFrame or pandas.Series.
            Exception: If method is called before transform() method.
        """
        # try in case plot_transformed() is called before transform()
        try:
            # if y is DataFrame use the first column to concat X_transformed and y
            if isinstance(y, pd.DataFrame):
                y = y[y.columns[0]]
                new_df = pd.concat([self.X_transformed, y], axis=1)

            # elif y is Series just concat X_transformed and y
            elif isinstance(y, pd.Series):
                new_df = pd.concat([self.X_transformed, y], axis=1)

            # else raise error because only DataFrame or Series can be used in pd.concat
            else:
                raise TypeError('Only pandas.DataFrame and pandas.Series object can be used as y.')

            # if n_conponents < 2 create sns.scatterplot
            if self.X_transformed.shape[1] < 3:
                plt.figure(figsize=figsize)
                sns.scatterplot(data=new_df, x=0, y=1, hue=y.name, **kwargs)

            # else create sns.pairplot to display all components
            else:
                sns.pairplot(new_df, hue=y.name, **kwargs)

        except AttributeError:
            raise Exception('Estimator was not created yet. Call transform() method.')

    def _init_methods(self):
        """
        Methods dictionary initialization.
        """
        self.methods_dict = {
            'pca': PCA,
            'svd': TruncatedSVD,
            'fa': FactorAnalysis,
            'nmf': NMF,
            'lda': LinearDiscriminantAnalysis,
            't_sne': TSNE,
            'isomap': Isomap,
            'lle': LocallyLinearEmbedding,
        }
