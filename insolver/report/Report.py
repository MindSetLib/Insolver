import builtins
import inspect

import pandas
from pandas_profiling import ProfileReport
from sklearn import metrics
from sklearn.utils.multiclass import type_of_target

import shutil
import jinja2


class Report:
    def __init__(self, model, task, X_train, y_train, predicted_train, X_test, y_test, predicted_test, directory='.'):
        """Class for making html report on results.

        Args:
            model: model-file
            task (str): "reg" or "class"
            X_train (:obj:`pandas.DataFrame`): Dataset with X_train.
            y_train (:obj:`pandas.Series`): Series with targets on train.
            predicted_train (:obj:`pandas.Series`): Series with predictions on train.
            X_test (:obj:`pandas.DataFrame`): Dataset with X_test.
            y_test (:obj:`pandas.Series`): Series with targets on test.
            predicted_test (:obj:`pandas.Series`): Series with predictions on test.
            directory (str): path to templates and where to save report.
        """
        self.model = model
        if task in ['reg', 'class']:
            self.task = task
        else:
            raise Exception(f"Not supported task class {task}")
        if (isinstance(X_train, pandas.DataFrame)
                and isinstance(X_test, pandas.DataFrame)
                and isinstance(y_train, pandas.Series)
                and isinstance(y_test, pandas.Series)
                and isinstance(predicted_train, pandas.Series)
                and isinstance(predicted_test, pandas.Series)):
            self.X_train = X_train
            self.y_train = y_train
            self.predicted_train = predicted_train
            self.X_test = X_test
            self.y_test = y_test
            self.predicted_test = predicted_test
        else:
            raise TypeError(f"""Wrong types of input data.
              \rX_train {type(X_train)} should be pandas.DataFrame
              \ry_train {type(y_train)} should be pandas.Series
              \rX_test {type(X_test)} should be pandas.DataFrame
              \ry_test {type(y_test)} should be pandas.Series
              \rpredicted_train {type(predicted_train)} should be pandas.Series
              \rpredicted_test {type(predicted_test)} should be pandas.Series
              \r""")
        self.directory = directory
        _directory = inspect.getfile(Report)
        self._directory = _directory[:_directory.rfind("/") + 1]

        self.profile = None
        self.profile_data()

        templateLoader = jinja2.FileSystemLoader(searchpath=self._directory)
        self.env = jinja2.Environment(loader=templateLoader)
        self.template = self.env.get_template("report_template.html")

        self.sections = [
                {
                  'name': 'Dataset',
                  'articles': [
                      {
                        'name': 'Pandas profiling',
                        'parts': [
                            '<div class="col-12"><button '
                            'class="btn btn-primary" type="submit" '
                            'onclick="window.location.href=\''
                            './profiling_report.html\';">'
                            'Go to report</button></div>'],
                        'header': '<p>Generated profile report from a '
                                  'pandas <code>DataFrame</code> prepared by '
                                  '<code>Pandas profiling library</code>.</p>',
                        'footer': '<a href="https://pypi.org/project/'
                                  'pandas-profiling/">library page</a>',
                      }
                   ],
                },
                {
                    'name': 'Model',
                    'articles': [
                            {
                                'name': 'Coefficients',
                                'parts': [self.model_features_importance()],
                                'header': '',
                                'footer': '',
                            },
                            {
                                'name': 'Metrics',
                                'parts': [self.calculate_train_test_metrics()],
                                'header': '',
                                'footer': '',
                            },
                            {
                                'name': 'Parameters',
                                'parts': self.model_parameters_to_list(),
                                'header': '',
                                'footer': '',
                            },
                        ],
                    },
                ]

    def profile_data(self):
        data_train = self.X_train.copy()
        data_train[self.y_train.name] = self.y_train
        data_test = self.X_test.copy()
        data_test[self.y_test.name] = self.y_train
        data = data_train.append(data_test)
        # Profiling
        self.profile = ProfileReport(data, title='Pandas Profiling Report')

    def model_features_importance(self):
        if self.model is not None:
            if self.model.backend in ["h2o", "sklearn"]:
                coefs = self.get_coefs_dict(self.model.coef_norm())
            elif self.model.backend in ['xgboost', 'lightgbm', 'catboost']:
                coefs = self.get_coefs_dict(self.model.shap(self.X_train.append(self.X_test), show=False))
            else:
                raise Exception("Unsupperted backend type {}".format(self.model.backend))

            coefs_head = ['relative_importance', 'scaled_importance', 'percentage']
            model_coefs = self.create_html_table(coefs_head, coefs, two_columns_table=False,
                                                 classes='table table-striped', justify='left')
        else:
            raise Exception("Model instance was not provided")
        return model_coefs

    def calculate_train_test_metrics(self):
        table_train = self.calc_metrics(self.y_train, self.predicted_train, self.task)
        table_test = self.calc_metrics(self.y_test, self.predicted_test, self.task)

        table = {key: [table_train.get(key, ''), table_test.get(key, '')] for key in table_train.keys()}
        model_metrics = self.create_html_table(["train", "test"], table, two_columns_table=False,
                                               classes='table table-striped', justify='left')
        return model_metrics

    def model_parameters_to_list(self):
        model_parameters_list = list()
        for table_name, table in self.get_objects_as_dicts(self.model):
            model_parameters_list.append(self.create_html_table([table_name], table, two_columns_table=True,
                                                                classes='table table-striped', justify='left'))
        return model_parameters_list

    def to_html(self):
        shutil.copytree(f'{self._directory}report_template', f'{self.directory}/report')
        self.profile.to_file(f"{self.directory}/report/profiling_report.html")

        with open(f'{self.directory}/report/report.html', 'w') as f:
            html_ = self.template.render(sections=self.sections)
            html_ = html_.replace('&#34;', '"').replace('&lt;', '<').replace('&gt;', '>')
            f.write(html_)

    def get_objects_as_dicts(self, obj: any, path='') -> list:
        """Method that saves any python object instances as dicts
        """
        def is_builtin(obj):
            if obj is None:
                return True
            else:
                return type(obj).__name__ in dir(builtins)

        result = list()

        if obj is None:
            return result
        elif type(obj) in [list, tuple]:
            for item in obj:
                result.extend(self.get_objects_as_dicts(item, path=f"{path}"))
        elif type(obj) in [dict]:
            for key, value in obj.items():
                result.extend(self.get_objects_as_dicts(value, path=f'{path}/{key}'))
        elif not is_builtin(obj) and '__dict__' in dir(obj):
            if obj.__dict__:
                result.append(("{}/{}".format(path, str(obj.__class__).replace('<', '').replace('>', '')),
                               {key: value for key, value in obj.__dict__.items() if key[0] != '_'}))
                result.extend(self.get_objects_as_dicts(obj.__dict__))
        return result

    @staticmethod
    def create_html_table(head: list, body: dict, two_columns_table: bool = False, **kwargs) -> str:
        """Create html code for table based on python dict instance

        Args:
            head (list): Columns' names.
            body (dict): Data for table, where dict keys are index names and dict values are data.
            two_columns_table (bool): Whether to store all data in one column or try to sparse data by columns.
             If True body values should have len(body['key']) == len(head) otherwise raised Exception.
            **kwargs (dict): Arguments passed to pandas.DataFrame.to_html(**kwargs) method.

        Returns:
            (str) HTML-code for table.

        """
        def check_body(x: dict):
            """Check if dict values are lists and all lists have same length."""
            for index, value in enumerate(body.values()):
                if index == 0:
                    if isinstance(value, list):
                        value_len_prev = len(value)
                    else:
                        return False
                elif not (isinstance(value, list) and len(value) == value_len_prev):
                    return False
            return True

        def check_head(head, body):
            """Checks if head list have same length with body values lists."""
            len_body_value = len(list(body.values())[0])

            if len_body_value != len(head):
                raise Exception(f"column names list length {len(head)} not equal to columns quantity {len_body_value}.")

        if not check_body(body) or two_columns_table:
            body = {key: [value] for key, value in body.items()}
        check_head(head, body)

        return pandas.DataFrame(data=body.values(), columns=head, index=body.keys()).to_html(**kwargs)

    @staticmethod
    def calc_metrics(true, predicted, task):
        """Function to calculate metrics."""
        result = dict()

        metrics_regression = {
            'explained_variance_score': metrics.explained_variance_score,
            'max_error': metrics.max_error,
            'mean_absolute_error': metrics.mean_absolute_error,
            'mean_squared_error': metrics.mean_squared_error,
            'median_absolute_error': metrics.median_absolute_error,
            'mean_absolute_percentage_error': metrics.mean_absolute_percentage_error,
            'r2_score': metrics.r2_score,
            'mean_poisson_deviance': metrics.mean_poisson_deviance,
            'mean_gamma_deviance': metrics.mean_gamma_deviance,
            'mean_tweedie_deviance': metrics.mean_tweedie_deviance,
            'd2_tweedie_score': metrics.d2_tweedie_score,
            'mean_pinball_loss': metrics.mean_pinball_loss,
        }
        metrics_classification = {
            "accuracy_score": metrics.accuracy_score,
            "average_precision_score": metrics.average_precision_score,
            "balanced_accuracy_score": metrics.balanced_accuracy_score,
            "brier_score_loss": metrics.brier_score_loss,
            "classification_report": metrics.classification_report,
            "cohen_kappa_score": metrics.cohen_kappa_score,
            "confusion_matrix": metrics.confusion_matrix,
            "det_curve": metrics.det_curve,
            "f1_score": metrics.f1_score,
            "hamming_loss": metrics.hamming_loss,
            "hinge_loss": metrics.hinge_loss,
            "jaccard_score": metrics.jaccard_score,
            "log_loss": metrics.log_loss,
            "matthews_corrcoef": metrics.matthews_corrcoef,
            "multilabel_confusion_matrix": metrics.multilabel_confusion_matrix,
            "precision_recall_curve": metrics.precision_recall_curve,
            "precision_recall_fscore_support": metrics.precision_recall_fscore_support,
            "precision_score": metrics.precision_score,
            "recall_score": metrics.recall_score,
            "roc_auc_score": metrics.roc_auc_score,
            "roc_curve": metrics.roc_curve,
            "zero_one_loss": metrics.zero_one_loss,
        }

        if task == "reg":
            functions_names = metrics_regression.keys()
            functions = metrics_regression
        elif task == "class":
            # type_of_target can be continuous or binary
            type_of_true = type_of_target(true)
            type_of_pred = type_of_target(predicted)
            functions = metrics_classification

            if type_of_true == 'binary' and type_of_pred == 'binary':
                functions_names = metrics_classification.keys()
            elif type_of_true == 'binary' and type_of_pred == 'continuous':
                functions_names = [
                                    "average_precision_score",
                                    "brier_score_loss",
                                    "det_curve",
                                    "hinge_loss",
                                    "log_loss",
                                    "precision_recall_curve",
                                    "roc_auc_score",
                                    "roc_curve",
                                    ]
            else:
                raise Exception(f"Not supported target type <{type_of_true}> or predicted type <{type_of_pred}>.")
        else:
            raise Exception(f"Not supported task type <{task}>. Currently supported {['class', 'reg']}.")

        for name in functions_names:
            try:
                result[name] = functions[name](true, predicted)
            except Exception as e:
                print(f'\t-{e}')
        return result

    @staticmethod
    def get_coefs_dict(model_coefs: dict) -> dict:
        """Extract InsolverGLMWrapper coef table ('relative_importance')
        with ['scaled_importance', 'percentage'] as h2o.
        """
        # create relative_importance value
        coefs = {key: [abs(value)] for key, value in model_coefs.items()}
        # exclude bias
        if coefs.get('Intercept') is not None:
            coefs.pop('Intercept', None)

        rel_imp_max = max([x[0] for x in coefs.values()])
        rel_imp_sum = sum([x[0] for x in coefs.values()])
        # add scaled_importance and percentage values
        coefs = {key: value + [value[0] / rel_imp_max, value[0] / rel_imp_sum]
                 for key, value in coefs.items()}

        return {key: value for key, value in sorted(coefs.items(), key=lambda x: x[1][0], reverse=True)}
