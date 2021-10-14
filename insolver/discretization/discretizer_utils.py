from sklearn.preprocessing import KBinsDiscretizer


class EqualWidthDiscretizer:
    @staticmethod
    def _transform(X, n_bins):
        return KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')\
            .fit_transform(X)


class EqualFrequencyDiscretizer:
    @staticmethod
    def _transform(X, n_bins):
        return KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')\
            .fit_transform(X)


class KMeansDiscretizer:
    @staticmethod
    def _transform(X, n_bins):
        return KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans') \
            .fit_transform(X)
