from pandas import DataFrame


class TransformParamUselessGroup:
    """Groups all parameter's values with few data to one group.

    Parameters:
        column_param (str): Column name in InsolverDataFrame containing parameter.
        size_min (int): Minimum allowed number of records for each parameter value, 1000 by default.
        group_name: Name of the group for parameter's values with few data.
        inference (bool): Sign if the transformation is used for inference, False by default.
        param_useless (list): The list of useless values of the parameter, for inference only.
    """

    def __init__(self, column_param, size_min=1000, group_name=0, inference=False, param_useless=None, priority=1):
        self.priority = priority
        self.column_param = column_param
        self.size_min = size_min
        self.group_name = group_name
        self.inference = inference
        if inference and param_useless is not None:
            self.param_useless = param_useless
        else:
            self.param_useless = []

    @staticmethod
    def _param_useless_get(df, column_param, size_min):
        """Checks the amount of data for each parameter's value.

        Args:
            df: InsolverDataFrame to explore.
            column_param (str): Column name in InsolverDataFrame containing parameter.
            size_min (int): Minimum allowed number of records for each parameter's value, 1000 by default.

        Returns:
            list: List of parameter's values with few data.
        """
        param_size = DataFrame(df.groupby(column_param).size().reset_index(name='param_size'))
        param_useless = list(param_size[column_param].loc[param_size['param_size'] < size_min])
        return param_useless

    def __call__(self, df):
        if self.param_useless == list():
            self.param_useless = self._param_useless_get(df, self.column_param, self.size_min)
        df.loc[df[self.column_param].isin(self.param_useless), self.column_param] = self.group_name
        return df


class TransformParamSortFreq:
    """Gets sorted by claims' frequency parameter's values.

    Parameters:
        column_param (str): Column name in InsolverDataFrame containing parameter.
        column_param_sort_freq (str): Column name in InsolverDataFrame for sorted values of parameter,
            column type is integer.
        column_policies_count (str): Column name in InsolverDataFrame containing numbers of policies,
            column type is integer or float.
        column_claims_count (str): Column name in InsolverDataFrame containing numbers of claims,
            column type is integer or float.
        inference (bool): Sign if the transformation is used for inference, False by default.
        param_freq_dict (dict): The dictionary of sorted values of the parameter, for inference only.
    """

    def __init__(
        self,
        column_param,
        column_param_sort_freq,
        column_policies_count,
        column_claims_count,
        inference=False,
        param_freq_dict=None,
        priority=2,
    ):
        self.priority = priority
        self.column_param = column_param
        self.column_param_sort_freq = column_param_sort_freq
        self.column_policies_count = column_policies_count
        self.column_claims_count = column_claims_count
        self.param_freq = DataFrame
        self.inference = inference
        if inference and param_freq_dict is not None:
            self.param_freq_dict = param_freq_dict
        else:
            self.param_freq_dict = {}

    def __call__(self, df):
        if self.param_freq_dict == dict():
            self.param_freq = df.groupby([self.column_param]).sum()[
                [self.column_claims_count, self.column_policies_count]
            ]
            self.param_freq['freq'] = (
                self.param_freq[self.column_claims_count] / self.param_freq[self.column_policies_count]
            )
            keys = []
            values = []
            for i in enumerate(self.param_freq.sort_values('freq', ascending=False).index.values):
                keys.append(i[1])
                values.append(float(i[0]))
            self.param_freq_dict = dict(zip(keys, values))
        df[self.column_param_sort_freq] = df[self.column_param].map(self.param_freq_dict)
        return df


class TransformParamSortAC:
    """Gets sorted by claims' average sum parameter's values.

    Parameters:
        column_param (str): Column name in InsolverDataFrame containing parameter.
        column_param_sort_ac (str): Column name in InsolverDataFrame for sorted values of parameter,
            column type is integer.
        column_claims_count (str): Column name in InsolverDataFrame containing numbers of claims,
            column type is integer or float.
        column_claims_sum (str): Column name in InsolverDataFrame containing sums of claims,
            column type is integer or float.
        inference (bool): Sign if the transformation is used for inference, False by default.
        param_ac_dict (dict): The dictionary of sorted values of the parameter, for inference only.
    """

    def __init__(
        self,
        column_param,
        column_param_sort_ac,
        column_claims_count,
        column_claims_sum,
        inference=False,
        param_ac_dict=None,
        priority=2,
    ):
        self.priority = priority
        self.column_param = column_param
        self.column_param_sort_ac = column_param_sort_ac
        self.column_claims_count = column_claims_count
        self.column_claims_sum = column_claims_sum
        self.param_ac = DataFrame
        self.inference = inference
        if inference and param_ac_dict is not None:
            self.param_ac_dict = param_ac_dict
        else:
            self.param_ac_dict = {}

    def __call__(self, df):
        if self.param_ac_dict == dict():
            self.param_ac = df.groupby([self.column_param]).sum()[[self.column_claims_sum, self.column_claims_count]]
            self.param_ac['avg_claim'] = self.param_ac[self.column_claims_sum] / self.param_ac[self.column_claims_count]
            keys = []
            values = []
            for i in enumerate(self.param_ac.sort_values('avg_claim', ascending=False).index.values):
                keys.append(i[1])
                values.append(float(i[0]))
            self.param_ac_dict = dict(zip(keys, values))
        df[self.column_param_sort_ac] = df[self.column_param].map(self.param_ac_dict)
        return df
