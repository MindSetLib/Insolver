import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, ElasticNet
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, chi2, f_classif, f_regression
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler


class FeatureSelection:
    """Feature selection. Supports the following tasks: classification, regression, multiclass classification and
    multiclass multioutput classification.
    
    Note:
        The following specified methods can be used for each individual task:

        - for the **classification** problem Mutual information, F statistics, chi-squared test, Random Forest, Lasso or
          ElasticNet can be used;
        - for the **regression** problem Mutual information, F statistics, Random Forest, Lasso or ElasticNet can be
          used;
        - for the **multiclass classification** Random Forest, Lasso or ElasticNet can be used;
        - for the **multiclass multioutput classification** Random Forest can be used.

        Random Forest is used by default.
   
    Parameters:
        y_column (str): The name of the column to predict.
        task (str): A task for the model. Values `reg`, `class`, `multiclass` and `multiclass_multioutput` are
         supported.
        method (str): A technique to compute features importance. Values `random_forest`(default), `mutual_inf`, `chi2`,
         `f_statistic`, 'lasso' and 'elasticnet' are supported.
        permutation_importance (bool): Uses permutation feature importance, false is default.
        
    Attributes:
        new_dataframe (pandas.DataFrame): New dataframe with the selected features only.
        importances (list): A list of the importances created using selected method.
        model : A model for feature selection.
        permutation_model : Permutation model for feature selection.
        
    """
    
    def __init__(self, y_column, task, method='random_forest', permutation_importance=False):
        self.y_column = y_column
        self.task = task
        self.method = method
        self.permutation_importance = permutation_importance
        self.new_dataframe = pd.DataFrame()
        self.importances = []
        self.tasks_list = ['reg', 'class', 'multiclass', 'multiclass_multioutput']
       
    def create_model(self, df):
        """
        A method to create a model for feature selection using specified method. Random Forest is used by default.
        
        Parameters:
            df (pandas.Dataframe): The dataframe.
        
        Raises:
            ValueError: If there are null values in the dataframe.
            ValueError: If there are object columns in the dataframe.
            NotImplementedError: If self.method isn't supported with the task.
            
        """
        
        if not df.isnull().sum().sum() == 0:
            raise ValueError('All values in the dataframe must be not null.')
            
        if len([var for var in df.columns if df[var].dtype == 'object']) > 0:
            raise ValueError('All values in the dataframe must not be object.')
            
        self._init_methods_dict()
        self._init_importance_dict()
            
        if self.method not in self.methods_dict.keys():
            raise NotImplementedError(
                f'Task {self.task} does not support method "{self.method}".')

        self.x = df.drop([self.y_column], axis=1)
        self.y = df[self.y_column]
        
        self.model = self.methods_dict[self.method](self.x, self.y) 
        self.importances = self.importance_dict[self.method](self.model)
        
        if self.permutation_importance:
            self.create_permutation_importance()
         
    def create_permutation_importance(self, **kwargs):
        """A method for creating permutation importance for the features. This method will be automatically called if
        'permutation_importance' parameter was set to True. Features importances will be set to importances_mean from
        permutation_importance model.
        
        Note:
            This method can be called only after method 'create_model' has been called.
       
        Raises:
            Exception: Model was not created, self.x or self.importances was not initialized.
            Exception: Permutation importance was used with the method that doesn't implement class
                sklearn.base.BaseEstimator.
        
        """
        try:
            self.permutation_model = permutation_importance(self.model, self.x, self.y, **kwargs)
            self.importances = self.permutation_model.importances_mean
        
        except AttributeError:
            raise Exception('Model was not created yet.')
        except TypeError:
            raise Exception('Permutation importance can only be used with the estimator.')
        
    def create_new_dataset(self, threshold='mean'):
        """
        A method for creating new dataset. It uses threshold parameter to select features.
        
        Note:
            This method can be called only after method 'create_model' has been called.
            This method uses absolute numeric value of the importences during comparison with the threshold value.
        
        Parameters:
            threshold : The threshold value to use. It can be 'mean'(default), 'median' or numeric.
        
        Raises:
            Exception: Model was not created.
        
        """

        try: 
            df_scores = pd.DataFrame({'feature_name': self.x.columns, 'feature_score': self.importances})
                    
            if threshold == 'mean':
                self.threshold = df_scores['feature_score'].abs().mean()
                cols = df_scores[df_scores['feature_score'] > self.threshold]['feature_name']
                self.new_dataframe = pd.concat([self.x[cols], self.y], axis=1)
                
                return self.new_dataframe
                
            elif threshold == 'median':
                self.threshold = df_scores['feature_score'].abs().median()
                cols = df_scores[df_scores['feature_score'] > self.threshold]['feature_name']
                self.new_dataframe = pd.concat([self.x[cols], self.y], axis=1)
                
                return self.new_dataframe
            
            else:
                self.threshold = threshold
                cols = df_scores[df_scores['feature_score'].abs() > self.threshold]['feature_name']
                self.new_dataframe = pd.concat([self.x[cols], self.y], axis=1)
                
                return self.new_dataframe

        except AttributeError:
            raise Exception('Model was not created yet.')
        
    def plot_importance(self, figsize=(5, 5), importance_threshold=None):
        """
        A method for plotting feature importance using created model.
        
        Note:
            This method can be called only after method 'create_model' has 
            been called.
        
        Parameters:
            figsize (list): Figsize of the plot.
            importance_threshold (float): The threshold of importance by which
                the features will be plotted. 
        
        Raises:
            Exception: Model was not created, self.x or self.importances was 
                not initialized.
        
        """
        try:
            if len(self.importances.shape) > 1:
                n = 0
                for n_class in self.importances:
                    df_to_plot = pd.DataFrame({
                        'feature_name': self.x.columns,
                        'feature_score': n_class})
                    
                    if importance_threshold:
                        df_to_plot[df_to_plot['feature_score'] > importance_threshold].plot.barh(
                            x='feature_name', y='feature_score', figsize=figsize)

                    else:
                        df_to_plot.plot.barh(x='feature_name', y='feature_score', figsize=figsize)

                    plt.title(f'Model {self.method} class {self.model.classes_[n]} scores')
                    n += 1

            else:
                df_to_plot = pd.DataFrame({
                    'feature name': self.x.columns,
                    'feature score': self.importances
                })
                
                if importance_threshold:
                    df_to_plot[df_to_plot['feature score'] > importance_threshold].plot.barh(
                        x='feature name', y='feature score', figsize=figsize)

                else:
                    df_to_plot.plot.barh(x='feature name', y='feature score', figsize=figsize)
                
                plt.title(f'Model {self.method} features scores')
            
        except AttributeError:
            raise Exception('Model was not created yet.')
         
    def _init_methods_dict(self):
        """
        Non-public method for creating a methods dictionary.
        
        Raises:
            NotImplementedError: If self.task is not supported.
        
        """
        if self.task == 'class':
            self.methods_dict = {
                'mutual_inf': lambda x, y: mutual_info_classif(x, y),
                'chi2': lambda x, y: chi2(x, y),
                'f_statistic': lambda x, y: f_classif(x, y),
                'random_forest': lambda x, y: RandomForestClassifier(n_estimators=10).fit(x, y),
                'lasso': lambda x, y: LogisticRegression(penalty='l1',
                                                         solver='saga').fit(StandardScaler().fit_transform(x), y),
                'elasticnet': lambda x, y: LogisticRegression(penalty='elasticnet', l1_ratio=0.5,
                                                              solver='saga').fit(StandardScaler().fit_transform(x), y),
            }
            
        elif self.task == 'reg':
            self.methods_dict = {
                'mutual_inf': lambda x, y: mutual_info_regression(x, y),
                'f_statistic': lambda x, y: f_regression(x, y),
                'random_forest': lambda x, y: RandomForestRegressor(n_estimators=10).fit(x, y),
                'lasso': lambda x, y: Lasso().fit(StandardScaler().fit_transform(x), y),
                'elasticnet': lambda x, y: ElasticNet().fit(StandardScaler().fit_transform(x), y),
            } 
       
        elif self.task == 'multiclass':
            self.methods_dict = {
                'random_forest': lambda x, y: RandomForestRegressor(n_estimators=10).fit(x, y),
                'lasso': lambda x, y:
                LogisticRegression(penalty='l1', solver='saga',
                                   multi_class='multinomial').fit(StandardScaler().fit_transform(x), y),
                'elasticnet': lambda x, y:
                LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga', multi_class='multinomial')
                .fit(StandardScaler().fit_transform(x), y),
            }
        
        elif self.task == 'multiclass_multioutput':
            self.methods_dict = {
                'random_forest': lambda x, y: RandomForestRegressor(n_estimators=10).fit(x, y),
            }
        
        else:
            raise NotImplementedError(f'Value task must be one of the {self.tasks_list}')
    
    def _init_importance_dict(self):
        """Non-public method for creating an importance dictionary.
        
        """
        self.importance_dict = {
            'random_forest': lambda model: model.feature_importances_,
            'mutual_inf': lambda model: model,
            'chi2': lambda model: model[1],
            'f_statistic': lambda model: -np.log10(model[1])/(-np.log10(model[1])).max(),
            'lasso': lambda model: model.coef_[0] if self.task == 'class' else model.coef_,
            'elasticnet': lambda model: model.coef_[0] if self.task == 'class' else model.coef_
        }
    
    def __call__(self, df):
        self.create_model(df)
        self.plot_importance()