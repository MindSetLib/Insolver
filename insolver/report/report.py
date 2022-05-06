import builtins
import glob
from pathlib import Path

import pandas
from pandas_profiling import ProfileReport
from sklearn import metrics
from sklearn.utils.multiclass import type_of_target

import presets
import metrics
import comparison_presets

import shutil
import jinja2


class Report:
    """Combine data and model summary in one html report
    
    Parameters:
        model (InsolverBaseWrapper): A fitted model implementing `predict`.
        task (str): Model type, supported values are `reg` and `class`.
        X_train (pandas.DataFrame): Train data.
        y_train (pandas.Series): Train target.
        predicted_train (pandas.Series): Train values predicted by the model.
        X_test (pandas.DataFrame): Test data.
        y_test (pandas.Series): Test target.
        predicted_test (pandas.Series): Test values predicted by the model.
        exposure_column (pandas.Series, str): Exposure column name for the gini coef and gain curve.
        dataset_description (str): Description of the dataset set to display.
        y_description (str): Description of the y value set to display.
        features_description (str): Features description set to display.
        metrics_to_calc (list): The names of the metrics to be calculated, can be `all` (all metrics will be 
            calculated), `main` or list.
        models_to_compare (list): Fitted models implementing `predict` for comparison.
        comparison_metrics (list): Metrics for comparison.
        f_groups_type (str, dict): Groups type for the `Features comparison chart`, supported values are: `cut` - bin 
            values into discrete intervals, `qcut` - quantile-based discretization function, `freq` - bins created 
            using start, end and the length of each interval. If str, all features are cut using `f_groups_type`. If 
            dict, must be {'feature': 'groups_type', 'all': 'groups_type'} where 'all' will be used for all features 
            not listed in the dict.
        f_bins (int, dict): Bins for the `Features comparison chart`. Number of bins for `cut` and `qcut` groups_type.
            If int, all features are cut using `f_bins`. If dict, must be {'feature': bins, 'all': 'groups_type'} 
            where 'all' will be used for all features not listed in the dict. Default value is 10.
        f_start (float, dict): Start for the `Features comparison chart`. Start value for `freq` groups_type. If not
            set, min(column)-1 is used. If float, all features are cut using `f_start`. If dict, must be 
            {'feature': start, 'all': 'groups_type'} where 'all' will be used for all features not listed in the dict.
        f_end (float, dict): End for the `Features comparison chart`. End value for `freq` groups_type. If not
            set, max(column)+1 is used. If float, all features are cut using `f_end`. If dict, must be 
            {'feature': end, 'all': 'groups_type'} where 'all' will be used for all features not listed in the dict.
        f_freq (float, dict): Freq for the `Features comparison chart`. The length of each interval for `freq` 
            groups_type. Default value is 1.5. If float, all features are cut using `f_freq`. If dict, must be 
            {'feature': freq, 'all': 'groups_type'} where 'all' will be used for all features not listed in the dict.
        p_groups_type (str): Groups type for the `Predict groups chart`, supported values are: `cut` - bin 
            values into discrete intervals, `qcut` - quantile-based discretization function, `freq` - bins created 
            using start, end and the length of each interval. 
        p_bins (int): Bins for the `Predict groups chart`. Number of bins for `cut` and `qcut` groups_type. Default 
            value is 10.
        p_start (float): Start for the `Predict groups chart`. Start value for `freq` groups_type. If not
            set, min(column)-1 is used.
        p_end (float): End for the `Predict groups chart`. End value for `freq` groups_type. If not
            set, max(column)+1 is used. 
        p_freq (float): Freq for the `Predict groups chart`. The length of each interval for `freq` 
            groups_type. Default value is 1.5. 
        d_groups_type (str): Groups type for the `Difference chart`, supported values are: `cut` - bin 
            values into discrete intervals, `qcut` - quantile-based discretization function, `freq` - bins created 
            using start, end and the length of each interval. 
        d_bins (int): Bins for the `Difference chart`. Number of bins for `cut` and `qcut` groups_type. Default 
            value is 10.
        d_start (float): Start for the `Difference chart`. Start value for `freq` groups_type. If not set, 
            min(column)-1 is used.
        d_end (float): End for the `Difference chart`. End value for `freq` groups_type. If not set, max(column)+1 
            is used. 
        d_freq (float): Freq for the `Difference chart`. The length of each interval for `freq` groups_type. 
            Default value is 1.5. 
        main_diff_model: Main difference model for the `Difference chart`.
        compare_diff_models (list): Models for comparison with the main model for the `Difference chart`. 
        pairs_for_matrix (list): List of pairs for the `Comparison matrix`.
        m_bins (int): Number of bins for the `Comparison matrix`.
        m_freq (float): The length of each interval for the `Comparison matrix`. If set, m_bins won't be used.

    Public methods:
        to_html(path, report_name): Generates html report.
        get_sections() Get created self.sections dict.

    """

    def __init__(self, model, task,
                 X_train, y_train,  
                 X_test, y_test, original_dataset,
                 predicted_train=None, predicted_test=None,
                 exposure_column=None,
                 dataset_description: str = 'Add a model description to the `dataset_description` parameter.',
                 y_description: str = 'Add a y description to the `y_description` parameter.',
                 features_description=None,
                 metrics_to_calc='main', models_to_compare=None, comparison_metrics=None,
                 f_groups_type='cut', f_bins=10, f_start=None, f_end=None, f_freq=1.5,
                 p_groups_type='cut', p_bins=10, p_start=None, p_end=None, p_freq=1.5,
                 d_groups_type='cut', d_bins=10, d_start=None, d_end=None, d_freq=1.5,
                 main_diff_model=None, compare_diff_models=None,
                 pairs_for_matrix=None, m_bins=20, m_freq=None,
                 show_parameters=False):
        # check and save attributes
        self.metrics_to_calc = metrics_to_calc
        self.exposure_column = exposure_column.name if isinstance(exposure_column, pandas.Series) else exposure_column
        self.model = model
        self.models_to_compare = models_to_compare
        self.comparison_metrics = [] if not comparison_metrics else comparison_metrics
        self.predicted_train = pandas.Series(model.predict(X_train),
                                             index=X_train.index) if not predicted_test else predicted_train
        self.predicted_test = pandas.Series(model.predict(X_test),
                                            index=X_test.index) if not predicted_train else predicted_test
        if task in ['reg', 'class']:
            self.task = task
        else:
            raise ValueError(f"Not supported task class {task}")
        if (isinstance(X_train, pandas.DataFrame)
                and isinstance(X_test, pandas.DataFrame)
                and isinstance(y_train, pandas.Series)
                and isinstance(y_test, pandas.Series)
                and isinstance(self.predicted_train, pandas.Series)
                and isinstance(self.predicted_test, pandas.Series)):
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
        else:
            raise TypeError(f"""Wrong types of input data.
              \rX_train {type(X_train)} must be pandas.DataFrame
              \ry_train {type(y_train)} must be pandas.Series
              \rX_test {type(X_test)} must be pandas.DataFrame
              \ry_test {type(y_test)} should be pandas.Series
              \rpredicted_train {type(self.predicted_train)} must be pandas.Series
              \rpredicted_test {type(self.predicted_test)} must be pandas.Series
              \r""")
        self._directory = Path().absolute()

        # prepare profile report
        self.profile = None
        self._profile_data()

        # prepare jinja environment and template
        templateLoader = jinja2.FileSystemLoader(searchpath=self._directory)
        self.env = jinja2.Environment(loader=templateLoader)
        self.template = self.env.get_template("report_template.html")
        
        # get features importance
        model_features_importance = self._model_features_importance()
        # calculate train test metrics
        calculate_train_test_metrics = self._calculate_train_test_metrics()
        # create lift chart and gain curve
        metrics_footer, metrics_part = metrics._create_metrics_charts(X_train, X_test,
                                                                      y_train, y_test, 
                                                                      self.predicted_train, self.predicted_test,
                                                                      exposure_column)
        # create partial dependence 
        pdp_footer, pdp_part = presets._create_partial_dependence(X_train, X_test, model)
        
        # content to fill jinja template
        self.sections = [
                {
                  'name': 'Dataset',
                  'articles': [
                      presets._create_dataset_description(X_train, X_test, y_train, y_test, task,
                                                          dataset_description, y_description,
                                                          original_dataset),
                      presets._create_pandas_profiling(),
                   ],
                  'icon': '<i class="bi bi-bricks" width="24" height="24" role="img"></i>',
                },
                {
                  'name': 'Model',
                  'articles': [
                      {
                        'name': 'Coefficients',
                        'parts': [f'''
                        <div class="p-3 m-3 bg-light border rounded-3 fw-light">
                            {model_features_importance[0]}{presets._create_importance_charts()}</div>'''],
                        'header': '',
                        'footer': model_features_importance[1],
                        'icon': '<i class="bi bi-bar-chart-line"></i>',
                          
                      },
                      {
                        'name': 'Metrics',
                        'parts': [f'{calculate_train_test_metrics[0]}{metrics_part}'],
                        'header': '',
                        'footer': metrics_footer,
                        'icon': '<i class="bi bi-calculator"></i>',
                      },
                      {
                        'name': 'Partial Dependence',
                        'parts': [f'''
                        <div class="p-3 m-3 bg-light border rounded-3 text-center fw-light">
                            {pdp_part}</div>'''],
                        'header': '',
                        'footer': pdp_footer,
                        'icon': '<i class="bi bi-graph-up"></i>',
                          
                      },
                   ],
                  'icon': '<i class="bi bi-tools"></i>',
                },
             ]
        
        # create features description article, contains specification, description and psi
        self.sections[0]['articles'].append(presets._create_features_description(X_train, X_test, 
                                                                                 original_dataset,
                                                                                 features_description))
        # create models comparison if model is regression
        if models_to_compare and task == 'reg':
            self.sections.append(
                comparison_presets._create_models_comparison(X_train, y_train, X_test, y_test, original_dataset, task,
                                                             models_to_compare, comparison_metrics, f_groups_type,
                                                             f_bins, f_start, f_end, f_freq, p_groups_type, p_bins,
                                                             p_start, p_end, p_freq, d_groups_type, d_bins, d_start,
                                                             d_end, d_freq, model, main_diff_model, compare_diff_models,
                                                             m_bins, m_freq, pairs_for_matrix,
                                                             classes="table table-striped",
                                                             justify="center"))
        # show all model parameters, some models have a lot of parameters, so they are not shown by default
        if show_parameters:
            self.sections[1]['articles'].append({
                        'name': 'Parameters',
                        'parts': self._model_parameters_to_list(),
                        'header': '',
                        'footer': '',
                        'icon': '<i class="bi bi-layout-text-sidebar-reverse"></i>',
            })

    def get_sections(self):
        return self.sections
    
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
                                or x.strip(f'{path_}/')[len(name_):None] == '')]

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
        # save profile report
        self.profile.to_file(f"{path}/{report_name}/profiling_report.html")

        with open(f'{path}/{report_name}/report.html', 'w') as f:
            html_ = self.template.render(sections=self.sections)
            html_ = html_.replace('&#34;', '"').replace('&lt;', '<').replace('&gt;', '>')
            f.write(html_)

    def _profile_data(self):
        """Combine all data passed in __init__ method and prepares report"""

        # train and test datasets into full dataset
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
        table_train = metrics._calc_metrics(self.y_train, self.predicted_train, self.task, self.metrics_to_calc,
                                            self.X_train, self.exposure_column)
        table_test = metrics._calc_metrics(self.y_test, self.predicted_test, self.task, self.metrics_to_calc,
                                           self.X_test, self.exposure_column)

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
                                                                     classes='table table-striped', justify='left')[0])
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
        
        result_df = pandas.DataFrame(data=body.values(), columns=head, index=body.keys())
        
        return [result_df.to_html(**kwargs),
                {'columns': head, 'data': [result_df[column].to_list() for column in result_df.columns],
                'index': list(result_df.axes[0])}]

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
