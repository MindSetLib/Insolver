import builtins
import inspect
import glob

import pandas
from pandas_profiling import ProfileReport
from sklearn import metrics
from sklearn.utils.multiclass import type_of_target

import shutil
import jinja2


class Report:
    """Combine data and model summary in one html report

    Public methods:
        to_html(): generates html report
    """

    def __init__(self, model, task, X_train, y_train, predicted_train, X_test, y_test, predicted_test):
        """
        Args:
            model (InsolverBaseWrapper): model-file.
            task (str): {"reg" or "class"}
            X_train (pandas.DataFrame): Data used to train model.
            y_train (pandas.Series): Series with targets on train.
            predicted_train (pandas.Series): Model response on train data.
            X_test (pandas.DataFrame): Data used to test model.
            y_test (pandas.Series): Series with targets on test.
            predicted_test (pandas.Series): Model response on test data.

        """
        # check and save attributes
        self.model = model
        if task in ['reg', 'class']:
            self.task = task
        else:
            raise ValueError(f"Not supported task class {task}")
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
        _directory = inspect.getfile(Report)
        self._directory = _directory[:_directory.rfind("/")]

        # prepare profile report
        self.profile = None
        self._profile_data()

        # prepare jinja environment and template
        templateLoader = jinja2.FileSystemLoader(searchpath=self._directory)
        self.env = jinja2.Environment(loader=templateLoader)
        self.template = self.env.get_template("report_template.html")

        # content to fill jinja template
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
                        'parts': [self._model_features_importance()],
                        'header': '',
                        'footer': '',
                      },
                      {
                        'name': 'Metrics',
                        'parts': [self._calculate_train_test_metrics()],
                        'header': '',
                        'footer': '',
                      },
                      {
                        'name': 'Parameters',
                        'parts': self._model_parameters_to_list(),
                        'header': '',
                        'footer': '',
                      },
                   ],
                },
             ]

    def to_html(self, path: str = '.', report_name: str = 'report'):
        """Saves prepared report to html file

        Args:
            path: existing location to save report
            report_name: name of report directory
        """
        def check_name(name_, path_):
            """Add a number to {name_} if it exists in {path_} directory"""

            check_names = [x.strip(f'{path_}/') for x in glob.glob(f"{path_}/*")
                           if x.strip(f'{path_}/').find(name_) == 0
                           and (x.strip(f'{path_}/')[len(name_):None].isnumeric()
                                or x.strip(f'{path_}/')[len(name_):None] == ''
                                )
                           ]

            name_to_check = name_
            name_count = len(check_names)
            while name_to_check in check_names:
                name_to_check = name_ + str(name_count)
                name_count += 1
            return name_to_check

        path = '.' if path == '' else path
        report_name = check_name(report_name, path)

        # copy template
        shutil.copytree(f'{self._directory}/report_template',
                        f'{path}/{report_name}')
        self.profile.to_file(f"{path}/{report_name}/profiling_report.html")

        with open(f'{path}/{report_name}/report.html', 'w') as f:
            html_ = self.template.render(sections=self.sections)
            html_ = html_.replace('&#34;', '"').replace('&lt;', '<').replace('&gt;', '>')
            f.write(html_)

    def _profile_data(self):
        """Combine all data passed in __init__ method and prepares report"""

        data_train = self.X_train.copy()
        data_train[self.y_train.name] = self.y_train
        data_test = self.X_test.copy()
        data_test[self.y_test.name] = self.y_train
        data = data_train.append(data_test)
        # Profiling
        self.profile = ProfileReport(data, title='Pandas Profiling Report')

    def _model_features_importance(self):
        """Depend on model backend prepare features importance list.

        Return:
            str: html table with features sorted by importance
        """
        if self.model is not None:
            if self.model.algo == "rf":
                coefs = self._get_coefs_dict({key: value for key, value in zip(self.model.model.feature_name_,
                                                                               self.model.model.feature_importances_)})
            elif self.model.algo == "glm":
                coefs = self._get_coefs_dict(self.model.coef_norm())
            elif self.model.algo == "gbm":
                coefs = self._get_coefs_dict(
                                self.model.shap(
                                    self.X_train.append(self.X_test),
                                    show=False))
            else:
                raise Exception("Unsupperted backend type {}".format(self.model.backend))

            coefs_head = ['relative_importance', 'scaled_importance', 'percentage']
            model_coefs = self._create_html_table(coefs_head, coefs, two_columns_table=False,
                                                  classes='table table-striped', justify='left')
        else:
            raise Exception("Model instance was not provided")
        return model_coefs

    def _calculate_train_test_metrics(self):
        table_train = self._calc_metrics(self.y_train, self.predicted_train, self.task)
        table_test = self._calc_metrics(self.y_test, self.predicted_test, self.task)

        table = {key: [table_train.get(key, ''), table_test.get(key, '')] for key in table_train.keys()}
        model_metrics = self._create_html_table(["train", "test"], table, two_columns_table=False,
                                                classes='table table-striped', justify='left')
        return model_metrics

    def _model_parameters_to_list(self):
        """Model parameters as html tables in one list"""
        model_parameters_list = list()
        for table_name, table in self._get_objects_as_dicts(self.model):
            if table:
                model_parameters_list.append(self._create_html_table([str(table_name)], table, two_columns_table=True,
                                                                     classes='table table-striped', justify='left'))
        return model_parameters_list

    def _get_objects_as_dicts(self, obj, path='') -> list:
        """Method that saves any python object instances as dict.

        Args:
            obj (any): Any type of python object.
            path (str): location of object inside original object.

        Returns:
            list: tuples like (<str: path>, <dict: object content>)
        """
        def is_builtin(obj):
            return True if obj is None else type(obj).__name__ in dir(builtins)

        result = list()
        if path.count('/') > 10:
            return result
        elif type(obj) in [list, tuple]:
            for item in obj:
                result.extend(self._get_objects_as_dicts(item, path=f"{path}"))
        elif type(obj) in [dict]:
            for key, value in obj.items():
                result.extend(self._get_objects_as_dicts(value, path=f'{path}/{key}'))
        elif not is_builtin(obj) and '__dict__' in dir(obj):
            if obj.__dict__:
                obj_dict = {key: value for key, value in obj.__dict__.items()
                            if key[0] != '_'}
                if obj_dict:
                    result.append(("{}/{}".format(path,
                                                  str(obj.__class__).replace('<', '').replace('>', '')), obj_dict))
            result.extend(self._get_objects_as_dicts(obj.__dict__, path))
        return result

    @staticmethod
    def _create_html_table(head: list, body: dict, two_columns_table: bool = False, **kwargs) -> str:
        """Create html table based on python dict instance

        Args:
            head (list): Column's names.
            body (dict): Data for table where dict keys are index names and dict values are data.
            two_columns_table (bool): whether to store all data in one column  or try to sparse data by columns.
             If True body values sholud have len(body['key']) == len(head) otherwise Exception is raised.
            **kwargs: arguments passed to DataFrame.to_html(**kwargs) method.

        Returns:
            str: html-code for table.
        """

        def check_body(body: dict):
            """Check if dict values are lists and have same length"""

            for index, value in enumerate(body.values()):
                if index == 0:
                    if isinstance(value, list):
                        value_len_prev = len(value)
                    else:
                        return False
                elif not (isinstance(value, list) and len(value) == value_len_prev):
                    return False
                value_len_prev = len(value)
            return True

        def check_head(head: list, body: dict):
            """Checks if head list have same length with body values lists."""

            len_body_value = len(list(body.values())[0])
            if len_body_value != len(head):
                raise Exception(f"column names list length {len(head)} not "
                                f"equal to columns quantity {len_body_value}")

        if not check_body(body) or two_columns_table:
            body = {key: [value] for key, value in body.items()}
        check_head(head, body)

        return pandas.DataFrame(data=body.values(), columns=head, index=body.keys()).to_html(**kwargs)

    @staticmethod
    def _calc_metrics(y_true, y_pred, task):
        """Function to calculate metrics

        Args:
            y_true (1d array-like, or label indicator array / sparse matrix): Ground truth (correct) target values.
            y_pred (1d array-like, or label indicator array / sparse matrix): Estimated targets as returned by an
             estimator.

        Returns:
            dict: Where keys are metrics' names and values are scores.
        """
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
            type_of_true = type_of_target(y_true)
            type_of_pred = type_of_target(y_pred)
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
                raise TypeError(f"Not supported target type <{type_of_true}> or predicted type <{type_of_pred}>")
        else:
            raise TypeError(f"Not supported task type <{task}>. Currently supported {['class', 'reg']}")

        for name in functions_names:
            try:
                result[name] = functions[name](y_true, y_pred)
            except Exception as e:
                print(f'\t-{e}')
        return result

    @staticmethod
    def _get_coefs_dict(model_coefs: dict) -> dict:
        """Extern Insolver glm wraper coef table ('relative_importance')
           with ['scaled_importance', 'percentage'] as in h2o library
        """
        # create relative_importance value
        coefs = {key: [abs(value)] for key, value in model_coefs.items()}
        # exclude bias
        if coefs.get('Intercept') is not None:
            coefs.pop('Intercept', None)

        rel_imp_max = max([x[0] for x in coefs.values()])
        rel_imp_sum = sum([x[0] for x in coefs.values()])
        # add scaled_importance and percentage values
        coefs = {key: value + [value[0] / rel_imp_max, value[0] / rel_imp_sum] for key, value in coefs.items()}
        return {key: value for key, value in sorted(coefs.items(), key=lambda x: x[1][0], reverse=True)}