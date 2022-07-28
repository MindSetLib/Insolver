import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


class SklearnDiscretizer:
    @staticmethod
    def _transform(X, n_bins, method):
        """Apply discretizations from scikit-learn.

        Args:
            X: 1-D array, The data to be descretized.
            n_bins (int): The number of bins.
            method (string): The method used by scikit-learn's KBinsDiscretizer. Either 'uniform', 'quantile' or
            'kmeans'.

        Returns:
            1-D array, The transformed data.

        References:
            [1] https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html

        """
        return KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=method).fit_transform(X).reshape(-1)


class CARTDiscretizer:
    @staticmethod
    def _transform(X, y, min_samples_leaf=None, min_tree_depth=1, max_tree_depth=3):
        """Apply CART discretization.

        Args:
            X: 1-D array, The data to be descretized.
            y: 1-D array, The target values.
            min_samples_leaf(int): The minimum number of samples required to be at a leaf node. A split point at any
            depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left
            and right branches.
            This may have the effect of smoothing the model, especially in regression.
                If int, then consider min_samples_leaf as the minimum number.
                If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum
                number of samples for each node.
                If None, then min_samples_leaf implicitly set to 0.1.

        Returns:
            1-D array, The transformed data.

        References:
            [1] Liu, Huan, et al. "Discretization: An enabling technique." Data mining and knowledge discovery 6.4
            (2002): 393-423.

        """
        X = X.reshape(-1, 1)
        min_samples_leaf = 0.1 if min_samples_leaf is None else min_samples_leaf
        depths = range(min_tree_depth, max_tree_depth + 1)
        roc_auc_scores = []
        for tree_depth in depths:
            tree_model = DecisionTreeClassifier(max_depth=tree_depth, min_samples_leaf=min_samples_leaf)
            scores = cross_val_score(tree_model, X, y, cv=3, scoring='roc_auc')
            roc_auc_scores.append(np.mean(scores))

        best = depths[np.where(roc_auc_scores == np.max(roc_auc_scores))[0][0]]

        tree_model = DecisionTreeClassifier(max_depth=best, min_samples_leaf=min_samples_leaf)
        tree_model.fit(X, y)

        return tree_model.predict_proba(X)[:, 1]


class ChiMergeDiscretizer:
    def _transform(self, X, y, n_bins):
        """Apply ChiMerge discretization

        Args:
            X: 1-D array, The data to be descretized.
            y: 1-D array, The target values.
            n_bins(int): The number of bins.

        Returns:
            1-D array, The transformed data.

        References:
            [1] Kerber, Randy. "Chimerge: Discretization of numeric attributes." Proceedings of the tenth national
            conference on Artificial intelligence. 1992.
            Available from: https://www.aaai.org/Papers/AAAI/1992/AAAI92-019.pdf
        """
        binned = np.copy(X)
        intervals = self.__get_chimerge_intervals(X, y, n_bins)

        for i in range(len(intervals)):
            binned[(binned >= intervals[i][0]) & (binned <= intervals[i][1])] = i

        return binned

    @staticmethod
    def _get_new_intervals(intervals, min_chi_index):
        new_intervals = np.empty((len(intervals) - 1, 2))
        found = False
        i = 0
        k = 0
        while k < len(new_intervals):
            if not found and i == min_chi_index:
                t = np.concatenate((intervals[i], intervals[i + 1]))
                new_intervals[k] = np.array([min(t), max(t)])
                i += 2
            else:
                new_intervals[k] = intervals[i]
                i += 1
            k += 1
        return new_intervals

    @staticmethod
    def _get_chi(values, values_reversed, target, target_unique, intervals, i):
        left0 = np.argmax(values == intervals[i][0])
        left1 = np.argmax(values == intervals[i + 1][0])
        right0 = len(values_reversed) - np.argmax(values_reversed == intervals[i][1])
        right1 = len(values_reversed) - np.argmax(values_reversed == intervals[i + 1][1])
        interval_0 = target[left0:right0]
        interval_1 = target[left1:right1]
        a_1 = np.bincount(interval_0, minlength=len(target_unique))
        a_2 = np.bincount(interval_1, minlength=len(target_unique))
        r_1 = np.sum(a_1)
        r_2 = np.sum(a_2)
        c_j = np.sum([a_1, a_2], axis=0)
        n = np.sum(c_j)
        e_1j = r_1 * c_j / n
        e_2j = r_2 * c_j / n
        chi = np.power(a_1 - e_1j, 2) / e_1j + np.power(a_2 - e_2j, 2) / e_2j
        return np.sum(np.nan_to_num(chi))

    def _get_vals(self, chi, intervals):
        min_chi_index = np.where(chi == np.min(chi))[0][0]
        intervals = self._get_new_intervals(intervals, min_chi_index)
        idx = np.array([min_chi_index - 1, min_chi_index, min_chi_index + 1])
        idx = idx[(idx >= 0) & (idx <= len(chi) - 1)]
        chi = np.delete(chi, idx)
        return intervals, chi, idx

    def __get_chimerge_intervals(self, values, target, max_intervals):
        intervals = np.dstack((np.unique(values), np.unique(values)))[0]
        target = target[np.argsort(values)]
        target_unique = pd.unique(target)  # faster than np.unique
        values = np.sort(values)
        values_reversed = values[::-1]
        chi = np.empty(0)
        # initial calculation chi2 for single-values intervals
        for i in range(len(intervals) - 1):
            chi_ = self._get_chi(values, values_reversed, target, target_unique, intervals, i)
            chi = np.insert(chi, i, chi_)
        intervals, chi, idx = self._get_vals(chi, intervals)

        while len(intervals) > max_intervals:
            # consequent recalculation chi2 for changed intervals
            for i in idx[:-1]:
                chi_ = self._get_chi(values, values_reversed, target, target_unique, intervals, i)
                chi = np.insert(chi, i, chi_)
            intervals, chi, idx = self._get_vals(chi, intervals)

        return intervals
