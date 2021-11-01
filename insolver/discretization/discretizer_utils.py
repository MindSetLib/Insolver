import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


class KBinsDiscretizer:
    @staticmethod
    def _transform(X, n_bins, method):
        """Apply discretizations from scikit-learn.

        Args:
            X: 1-D array, The data to be descretized.
            n_bins (int): The number of bins.
            method (string): The method used by scikit-learn's KBinsDiscretizer. Either 'uniform', 'quantile' or 'kmeans'.

        Returns:
            1-D array, The transformed data.

        References:
            [1] https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html
        """
        return KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=method).fit_transform(X)


class CARTDiscretizer:
    @staticmethod
    def _transform(X, y, min_samples_leaf=None):
        """
        Apply CART discretization

        Args:
            X: 1-D array, data to be descretized
            y: 1-D array, target values
            min_samples_leaf: The minimum number of samples required to be at a leaf node. A split point at any depth will
            only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches.
            This may have the effect of smoothing the model, especially in regression.
                If int, then consider min_samples_leaf as the minimum number.
                If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
                If None, then min_samples_leaf implicitly set to 0.1

        Returns:
            1-D array, transformed data

        References:
            [1] Liu, Huan, et al. "Discretization: An enabling technique." Data mining and knowledge discovery 6.4 (2002): 393-423.
        """
        min_samples_leaf = 0.1 if min_samples_leaf is None else min_samples_leaf

        roc_auc_scores = []
        for tree_depth in range(1, 6):
            tree_model = DecisionTreeClassifier(max_depth=tree_depth, min_samples_leaf=min_samples_leaf)
            scores = cross_val_score(tree_model, X.to_frame(), y, cv=3, scoring='roc_auc')
            roc_auc_scores.append(np.mean(scores))

        best = np.where(roc_auc_scores == np.max(roc_auc_scores))[0][0]

        tree_model = DecisionTreeClassifier(max_depth=best, min_samples_leaf=min_samples_leaf)
        tree_model.fit(X.to_frame(), y)

        return tree_model.predict_proba(X.to_frame())[:, 1]
