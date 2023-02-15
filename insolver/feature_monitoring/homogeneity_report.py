import os
import inspect
import numpy as np
import pandas as pd
import plotly as py
import jinja2
from os.path import dirname
from typing import List, Sequence, Dict, Union
from plotly.figure_factory import create_distplot
from plotly import express as px

from .homogeneity_tests import ContinuousHomogeneityTests, DiscreteHomogeneityTests, fillna_cont, fillna_discr


def chart_cont(
    x1: np.ndarray, x2: np.ndarray, name1: str, name2: str, limits: Sequence, bins: int = 15, offline: bool = True
) -> py.graph_objs.Figure:
    """
    This function draws histograms of given samples using joint grid.
    It needs limits of interested area and number of bins.

    Parameters:
        x1 (np.array): sample from base period.
        x2 (np.array): sample from current period.
        name1 (str): name to describe base period.
        name2 (str): name to describe current period.
        limits (iterable of two floats or ints): min and max points to display distribution.
        bins (int): number of buckets to draw histogram.
        offline (bool): whether to return plot for rendering in template or to return figure itself.

    Returns:
        plotly offline plot with histograms in case when 'offline' parameter is 'True'
        else it returns plotly figure with histograms.
    """

    group_labels = [name1, name2]

    # drop outliers using limits
    x1_group = x1[(x1 >= limits[0]) & (x1 <= limits[1])]
    x2_group = x2[(x2 >= limits[0]) & (x2 <= limits[1])]

    # count min, max, size of bin
    min_ = min(np.min(x1_group), np.min(x2_group))
    max_ = max(np.max(x1_group), np.max(x2_group))
    bin_size = (max_ - min_) / bins

    # discretize values to get accurate histograms
    segments = (x1_group - min_) // bin_size
    x1_group = segments * bin_size + (bin_size / 2 + min_)

    segments = (x2_group - min_) // bin_size
    x2_group = segments * bin_size + (bin_size / 2 + min_)

    # draw hists
    hist_data = [x1_group, x2_group]
    fig = create_distplot(
        hist_data, group_labels, bin_size=bin_size, histnorm='probability', show_curve=False, show_rug=False
    )

    # add details
    fig.update_layout(
        autosize=False,
        width=830,
        height=650,
        xaxis_range=None,
        legend=dict(x=0.8, y=0.95, traceorder='normal', font=dict(color='black', size=16)),
    )
    if offline:
        return py.offline.plot(fig, include_plotlyjs=False, output_type='div')
    else:
        return fig


def chart_discr(x1: np.ndarray, x2: np.ndarray, name1: str, name2: str, offline: bool = True) -> py.graph_objs.Figure:
    """
    This function draws histograms of given samples using joint grid.
    It needs limits of interested area and number of bins.

    Parameters:
        x1 (np.array): sample from base period.
        x2 (np.array): sample from current period.
        name1 (str): name to describe base period.
        name2 (str): name to describe current period.
        offline (bool): whether to return plot for rendering in template or to return figure itself.

    Returns:
        plotly offline plot with histograms in case when 'offline' parameter is 'True'
        else it returns plotly figure with histograms.
    """

    # draw discrete hists
    fig1 = px.histogram(x1, histnorm='probability', barmode='overlay', color_discrete_sequence=['green'])
    fig1.for_each_trace(lambda t: t.update(name=name1))
    fig2 = px.histogram(x2, histnorm='probability', barmode='overlay', color_discrete_sequence=['red'])
    fig2.for_each_trace(lambda t: t.update(name=name2))
    fig = py.graph_objects.Figure(data=fig1.data + fig2.data)

    # add details
    fig.update_layout(
        autosize=False,
        width=830,
        height=650,
        legend=dict(x=0.8, y=0.95, traceorder='normal', font=dict(color='black', size=16)),
    )

    if offline:
        return py.offline.plot(fig, include_plotlyjs=False, output_type='div')
    else:
        return fig


class HomogeneityReport:
    """
    This class builds homogeneity report for two dataframes. Report consists of homogeneity checks
    between feature's condition in first and in second frame.
    We run statistical tests and draw joint charts with distributions if necessary.

    The class supports wide configuration of running tests. It takes config. dictionary which
    has feature names as keys. Each value for certain feature is sub-dictionary of properties.
    List of supported properties:
        feature_type (str): stat. type of features. ('continuous'/'discrete')
        pval_thresh (float): threshold for pvalue for making conclusions in tests.
        samp_size (int): size of sub-samples in bootstrap.
        bootstrap_num (int): number of bootstrap restarts.
        psi_bins (int): number of bins for psi calculation (only for 'continuous').
        chart_bins (int): number of bins to draw hists (only for 'continuous').
        chart_limits (iterable of two floats/ints): custom limits for continuous feature charts
        ('only for continuous').

    Only 'feature_type' ('continuous'/'discrete') must be specified for each feature,
    other properties can be missing.

    Default parameters for each property:
    pval_thresh=0.05, samp_size=500, bootstrap_num=100,
    psi_bins=20, chart_bins=15, chart_limits=[min(x1, x2), max(x1, x2)].

    Parameters:
        config_dict (dict): dict. with feature properties (see description).
    """

    def __init__(self, config_dict_inp: dict):
        # work with conf. dict as with property - all changes will go through setter and raise errors if necessary
        # calling setter:
        self.config_dict = config_dict_inp

    @property
    def features(self) -> List:
        return list(self.__config_dict.keys())

    @property
    def config_dict(self) -> Dict:
        return self.__config_dict

    @config_dict.setter
    def config_dict(self, config_dict_inp: Dict) -> None:
        """
        Raises:
            ValueError: if config_dict is empty. It must have some features.
            KeyError: if it is not specified whether certain feature is continuous or discrete.
            ValueError: if 'feature_type' is not 'continuous' or 'discrete'.
        """

        if config_dict_inp == {}:
            raise ValueError("Expected to get config with some features but not empty.")

        for feat in config_dict_inp:
            properties = config_dict_inp[feat]

            # check feature_type property
            if 'feature_type' not in properties:
                raise KeyError(f"Type of {feat} feature is not found in 'config_dict'.")
            elif properties['feature_type'] not in ['continuous', 'discrete']:
                raise ValueError(f"Types of features must be 'continuous' or 'discrete'. Invalid type for {feat}.")

        self.__config_dict = config_dict_inp

    def build_report(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        dropna: bool = False,
        name1: str = 'Base subset',
        name2: str = 'Current subset',
        draw_charts: bool = False,
    ) -> List:
        """
        Main function which assembles all testing logic - it takes raw dataframes
        and runs homogeneity tests for features. Feature set and properties are took from config dict.
        It can be specified to draw charts besides running tests.

        Parameters:
            df1 (pd.DataFrame): set of feature samples from base period.
            df2 (pd.DataFrame): set of feature samples from current period.
            dropna (bool): whether to do drop missing values or not while building report.
            name1 (str): name to describe base period.
            name2 (str): name to describe current period.
            draw_charts (bool): whether to render report or not.

        Returns:
            report_data (list): list of sub-lists. Each sublist contains 3 or 4 elements
            depending on 'draw_charts' param.
            1) string describing comparison
            2) results of each test with conclusion
            3) nan gap information - dict with percent of nans in each frame and difference between percents
            4) str with plotly offline plot with joint distribution plots (if 'draw_chart' == True)

        Raises:
            TypeError: if df1 is not pd.DataFrame.
            TypeError: if df2 is not pd.DataFrame.
            KeyError: if some feature are not found in df1.
            KeyError: if some feature are not found in df2.
            TypeError: if some feature don't have same dtype in df1 and df2.
            TypeError: if some feature has unsupported dtype in df1 or df2 (must be int, float or object).

            Warning: if 'psi_bins' is specified for discrete feature.
            Warning: if 'chart_bins' is specified for discrete feature.
            Warning: if 'chart_limits' if specified for discrete feature.
        """

        # checking error situations
        if not isinstance(df1, pd.DataFrame):
            raise TypeError("df1 must be a pandas DataFrame.")
        if not isinstance(df2, pd.DataFrame):
            raise TypeError("df2 must be a pandas DataFrame.")
        features = self.features
        if not (set(features) <= set(df1.columns)):
            raise KeyError("Can not find some features from configuration in df1.")
        if not (set(features) <= set(df2.columns)):
            raise KeyError("Can not find some features from configuration in df2.")

        # carefully assemble report data
        report_data = []
        for feat in features:
            properties = self.config_dict[feat]

            # required parameter
            feat_type = properties['feature_type']

            # check optional stat. parameters
            pval_thresh = 0.05 if ('pval_thresh' not in properties) else properties['pval_thresh']
            samp_size = 500 if ('samp_size' not in properties) else properties['samp_size']
            bootstrap_num = 100 if ('bootstrap_num' not in properties) else properties['bootstrap_num']

            # count nan difference between x1, x2
            nan_perc1 = df1[feat].isna().sum() / len(df1[feat])
            nan_perc2 = df2[feat].isna().sum() / len(df2[feat])
            nan_perc_gap = nan_perc2 - nan_perc1
            nan_gap_dict = {'nan_perc1': nan_perc1, 'nan_perc2': nan_perc2, 'nan_perc_gap': nan_perc_gap}

            # check data type errors
            if df1[feat].dtype != df2[feat].dtype:
                raise TypeError("All features must be of same data type.")

            if df1[feat].dtype not in [int, float, object]:
                raise TypeError("Only int, float or object datatypes are supported as for features.")

            # copy data to avoid side effects
            if dropna:
                x1 = df1[feat].dropna().values
                x2 = df2[feat].dropna().values
            else:
                x1 = df1[feat].values.copy()
                x2 = df2[feat].values.copy()

            if feat_type == 'continuous':
                # optional psi_bins
                psi_bins = 20 if ('psi_bins' not in properties) else properties['psi_bins']

                # manually fill nans
                x1, x2, _ = fillna_cont(x1, x2, inplace=True)

                # run tests
                homogen_tester: Union['ContinuousHomogeneityTests', 'DiscreteHomogeneityTests'] = (
                    ContinuousHomogeneityTests(pval_thresh, samp_size, bootstrap_num, psi_bins)
                )
                test_results = homogen_tester.run_all(x1, x2, inplace=True)

                # optional drawing of charts
                if draw_charts:
                    chart_bins = 15 if ('chart_bins' not in properties) else properties['chart_bins']
                    if 'chart_limits' not in properties:
                        chart_limits = min(np.min(x1), np.min(x2)), max(np.max(x2), np.max(x2))
                    else:
                        chart_limits = properties['chart_limits']

                    chart = chart_cont(x1, x2, name1, name2, chart_limits, chart_bins, offline=True)

            elif feat_type == 'discrete':
                # manually fill nans
                x1, x2, nan_value = fillna_discr(x1, x2, inplace=True)

                # run tests
                homogen_tester = DiscreteHomogeneityTests(pval_thresh, samp_size, bootstrap_num)
                test_results = homogen_tester.run_all(x1, x2, inplace=True)

                # optional drawing charts
                if draw_charts:
                    # ambiguous parameters for discrete feature
                    if 'psi_bins' in properties:
                        raise Warning(f"Ignoring 'psi_bins' argument for {feat} discrete feature.")
                    if 'chart_bins' in properties:
                        raise Warning(f"Ignoring 'chart_bins' argument for {feat} discrete feature.")
                    if 'chart_limits' in properties:
                        raise Warning(f"Ignoring 'chart_limits' argument for {feat} discrete feature.")

                    if x1.dtype == object:
                        x1, x2 = x1.astype(str), x2.astype(str)
                    else:
                        idx1 = x1 == nan_value
                        idx2 = x2 == nan_value
                        x1 = x1.astype(str)
                        x2 = x2.astype(str)
                        x1[idx1] = 'nan'
                        x2[idx2] = 'nan'
                    chart = chart_discr(x1, x2, name1, name2, offline=True)

            # reduce stat. results to format
            for i, result in enumerate(test_results):
                rep_dict = dict()
                rep_dict['test'], rep_dict['p_value'], rep_dict['conclusion'] = result
                test_results[i] = rep_dict

            # assemble data with charts or without them
            if draw_charts:
                feat_report = [f"{feat}: {name1} VS {name2}", test_results, nan_gap_dict, chart]
            else:
                feat_report = [f"{feat}: {name1} VS {name2}", test_results, nan_gap_dict]
            report_data.append(feat_report)

        return report_data


def render_report(report_data: list, report_path: str = 'homogeneity_report.html') -> None:
    """
    This is a separate function to render reports built by 'HomogeneityReport' class.
    Several report data lists can be concatenated and passed to this function.

    Parameters:
        report_data (list): list containing descriptions, results of tests, nan gaps and charts
        (see output of report_builder).
        report_path (str): path to save rendered report.

    Returns:
        None (as the result of the func. is html file on disk).

    Raises:
        OSError: if function didn't find 'report_template.html' file in insolver.
        KeyError: if dict of test results don't contain name of test, pvalue or conclusion.
        KeyError: if nan gap dict don't contain nan_perc1, nan_perc2 or nan_perc_gap.
    """

    # check template file
    curr_folder = dirname(inspect.getfile(HomogeneityReport))
    template_path = curr_folder + '/' + 'report_template.html'
    if not os.path.exists(template_path):
        raise OSError("Can not find template file. It must be in 'feature_monitoring' package.")

    # error situations
    for feat_report in report_data:
        for test_data in feat_report[1]:
            if ('test' not in test_data) or ('p_value' not in test_data) or ('conclusion' not in test_data):
                raise KeyError("Missing information in test data dict.")
        if (
            ('nan_perc1' not in feat_report[2])
            or ('nan_perc2' not in feat_report[2])
            or ('nan_perc_gap' not in feat_report[2])
        ):
            raise KeyError("Missing information in nan gap dict.")

    # render report
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(curr_folder))
    template = env.get_template("report_template.html")
    output = template.render(sets=report_data)

    with open(report_path, 'w') as f:
        f.write(output)
