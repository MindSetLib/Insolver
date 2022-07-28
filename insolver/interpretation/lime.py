import pandas as pd
import lime.lime_tabular as lt
from .base import InterpretBase


class LimeExplanation(InterpretBase):
    """
    Local Interpretable Model-Agnostic Explanations (lime). Uses lime package.

    Parameters:
        estimator: A fitted estimator object implementing `predict` or `predict_proba`.
        x (pandas.Dataframe, numpy.ndarray): Training data. If x is pandas.Dataframe, it is converted to the
            numpy.ndarray.
        mode (str): Type of the model. Values `classification` and `regression` are supported.
        feature_names (list): List of names corresponding to the columns in the training data.
        categorical_features (list): List of indices corresponding to the categorical columns. Everything else will be
            considered continuous. Values in these columns must be integers.
        kernel_width: Kernel width for the exponential kernel. If None, defaults to sqrt (number of columns) * 0.75.
        kernel: Similarity kernel that takes euclidean distances and kernel width as input and outputs weights in (0,1).
            If None, defaults to an exponential kernel.
        class_names (list): List of class names, ordered according to whatever the classifier is using.
    """

    def __init__(
        self,
        estimator,
        x,
        mode,
        feature_names,
        categorical_features=None,
        kernel_width=None,
        kernel=None,
        class_names=None,
    ):
        self.estimator = estimator
        self.x = x
        self.mode = mode
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.kernel_width = kernel_width
        self.kernel = kernel
        self.class_names = class_names

    def show_explanation(self, instance, result_type='show_in_notebook', file_path=None, label=None):
        """
        Show explanation. Creates lime.lime_tabular.LimeTabularExplainer and explains instance.

        Parameters:
            instance: Instance to explain.
            result_type (str): Type of the result. Values `html`, `list`, `map`, `file`, `show_in_notebook`
                are supported.
            file_path (str): File path, required if result_type is `file`.
            label (int, list): Desired labels to show explanations for.

        Raises:
            AttributeError: If label is not initialized and self.mode == 'classification' and result_type == 'list'.
        """

        # set prediction function as self.estimator.predict_proba if mode is classification
        if self.mode == 'classification':
            predict_fn = self.estimator.predict_proba
        # set prediction function as self.estimator.predict if mode is regression
        elif self.mode == 'regression':
            predict_fn = self.estimator.predict
        # raise error if mode is not supported
        else:
            raise NotImplementedError(f'Mode {self.mode} is not supported.')

        # initialize return type dictionary
        self._init_result_type_dict(label=label, file_path=file_path)

        # raise error if return type is not supported
        if result_type not in self.result_type_dict.keys():
            raise NotImplementedError(f'Return type {result_type} is not supported.')

        training_data = self.x.to_numpy() if isinstance(self.x, pd.DataFrame) else self.x

        # create lime.lime_tabular.LimeTabularExplainer
        self.model = lt.LimeTabularExplainer(
            training_data=training_data,
            mode=self.mode,
            feature_names=self.feature_names,
            categorical_features=self.categorical_features,
            kernel_width=self.kernel_width,
            kernel=self.kernel,
            class_names=self.class_names,
        )

        # explain instance using LimeTabularExplainer
        explanation = self.model.explain_instance(data_row=instance, predict_fn=predict_fn)

        if self.mode == 'classification' and result_type == 'list' and label is None:
            raise AttributeError('label must be initialized in classification if result_type is `list`.')

        # show explanation
        return self.result_type_dict[result_type](explanation)

    def plot(self, figsize, **kwargs):
        super.plot()

    def get_model(self):
        return self.model

    def _init_result_type_dict(self, label, file_path):
        self.result_type_dict = {
            'html': lambda explanation: explanation.as_html(labels=label),
            'list': lambda explanation: explanation.as_list(label=label),
            'map': lambda explanation: explanation.as_map(),
            'file': lambda explanation: explanation.save_to_file(file_path=file_path),
            'show_in_notebook': lambda explanation: explanation.show_in_notebook(labels=label),
        }
