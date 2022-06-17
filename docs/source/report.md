# Report
The report module allows you to create an HTML report containing information about your data and model.  

Report HTML file rendered by `jinja2` library, template of report file is based on bootstrap library template [link](https://getbootstrap.com/docs/5.1/examples/cheatsheet/).
## Report creation
To create a report, you need to have a trained model, a dataset and data splitted into train and test. Predicted variables can also be created, so there will be no need to predict train and test each time a `Report` instance is created.
Method `to_html(path, report_name)` creates a folder with generated files.

Example:
```python
from insolver.wrappers import InsolverRFWrapper, InsolverGBMWrapper, InsolverGLMWrapper
from insolver.report import Report
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('C:/your_df_location')

x_train, x_test, y_train, y_test = train_test_split(
    df.drop(['Y'], axis = 1), df['Y'])

irf = InsolverRFWrapper(backend='sklearn', task='reg')
irf.fit(x_train, y_train)
predict_rf_train = pd.Series(irf.predict(x_train), index=x_train.index)
predict_rf_test = pd.Series(irf.predict(x_test), index=x_test.index)

r = Report(model = model_name, task = 'reg',
           X_train = x_train, y_train = y_train,
           predicted_train = predict_rf_train,
           X_test = x_test, y_test = y_test,
           predicted_test = predict_rf_test,
           original_dataset = df
) 
r.to_html(report_name='random_forest_report')
```
## Report structure
Report template hierarchy:
```
Section 1:
   Article 1:
      part 1
      part 2
      part 3
   Article 2:
      part 1
      part 2
      part 3
Section 2:
   Article 1:
      part 1
      part 2
      part 3
   Article 2:
      part 1
      part 2
      part 3
```

In `insolver.report.Report` sections, articles, and parts are stored in the `sections` parameter of the Report class. `Article` is a `dict` containing the article's name, `list` of parts, header, and footer. Header and footer are optional fields where you can add comments before (header) and after (footer) content. Every `part` is a python string containing HTML code.

Method `get_sections()` returns `sections` parameter:
```python
from insolver.report import Report
r = Report(...) # sections are created at initialization
sections = r.get_sections()
```

## Report content
The rendered report has three sections, **Dataset**, **Model** and **Compare models**.

### Dataset section
The dataset section describes the dataset, its' features, target value and contains the pandas profiling file.
#### Dataset description article
This article contains the dataset specification from the `dataset_description` parameter, the train/test split info, the y specification the from `y_description` parameter, the y distibution chart and the y values description created using pandas.Dataframe.describe().
```python
from insolver.report import Report
dataset_descr = '''Some dataset description'''
y_descr = '''Score between 0 and 10.'''
r = Report(...,
           y_description=y_descr,
           dataset_description=dataset_descr) 
```
#### Pandas profiling article
This article contains a button with a link to the pandas profiling file. This file is created using [`pandas-profiling library`](https://github.com/pandas-profiling/pandas-profiling).

To generate a profiling report, all data passed to a Report class is combined and sent to the `pandas_profiling.ProfileReport` class.
#### Features description article
This article contains the features specification from the `features_description` parameter, the features values description created using `pandas.Dataframe.describe()` and the Population Stability Index (PSI).
```python
from insolver.report import Report
feat_descr = {
    'feat1': 'some text',
    'feat2': 'some text', ....
}
r = Report(...,
           features_description = feat_descr) 
```
### Model section
The model section describes feature importance coefficients, metrics, and model parameters.

#### Coefficients article
Coefficients are a numerical representation of features impact on predictions of the model, calculated depending on model type:
- **Random Forest**: coefficients from `RandomForest.feature_importance_` parameter.
- **Linear Model**: estimated coefficients for the linear model taken from `InsolverGMLWrapper.coef()` method.
- **Boosting Models**: SHAP interaction values computed for a boosting model.

#### Metrics article
Metrics are calculated depending on the type of task (`classification` or `regression`).
This article also contains the `Lift Chart` and the `Gain Curve`(if exposure_column is initialized).

### SHAP article
[SHAP (SHapley Additive exPlanations)](https://github.com/slundberg/shap) is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions.
This article contains:
**Mean SHAP values** chart with mean shap value for each feature;
***Feature* SHAP values** chart.

#### Partial Dependence article
A partial Dependence chart is generated for each feature with train\test selection. PDP is created using [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.PartialDependenceDisplay.html).

#### Explain instance article
This article is generated if the `explain_instance` parameter is pandas.Series. Instance is explained using SHAP values and [lime](https://github.com/marcotcr/lime) module.
This article contains **SHAP Waterfall chart** and **Lime chart**.

#### Parameters article
Inner state of model instance (parameters). Not shown by default.

### Compare models section
This section has two articles: Compare on train data and Compare on test data. They have the same content, with the difference that it is created using train and test data, respectively.
Compare models section is created if the `task` parameter is `reg` and the `models_to_compare` parameter is initialized. The `models_to_compare` parameter must be a list containing trained models. The `comparison_metrics` parameter is a list of metrics to add to the comparison.
```python
from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred): 
    return np.sqrt(mean_squared_error(y_true, y_pred))

irf_2 = InsolverRFWrapper(backend='sklearn', task='reg', n_estimators=10)
irf_2.fit(x_train, y_train)
iglm = InsolverGLMWrapper(backend='sklearn', family=0, standardize=True)
iglm.fit(x_train, y_train)
igbm = InsolverGBMWrapper(backend='catboost', task='reg', n_estimators=10)
igbm.fit(x_train, y_train)

r = Report(...,
           models_to_compare=[irf, igbm, iglm],
           comparison_metrics=[mean_squared_error, rmse]) 
```
**Metrics comparison table**
Models comparison is created using `insolver.models_tools.ModelMetricsCompare`. It returns a dataframe containing the metrics(from the `comparison_metrics` parameter) results for each model.

**Metrics comparison chart**
This is a bar that visualizes the metrics comparison table.

**Predict groups chart**
This chart is generated using prediction results for each model, grouped for each model with the option to choose between models.
Group creation parameters:
- `p_groups_type` (str): Supported values are: `cut` - bin values into discrete intervals, `qcut` - quantile-based discretization function, `freq` - bins created using start, end and the length of each interval. 
- `p_bins` (int): Number of bins for `cut` and `qcut` groups_type. Default value is 10.
- `p_start` (float): Start value for `freq` groups_type. If not set, min(column)-1 is used.
- `p_end` (float): End value for `freq` groups_type. If not set, max(column) is used. 
- `p_freq` (float): The length of each interval for `freq` groups_type. Default value is 1.5. 

**Difference chart**
A difference chart shows the difference between the main model and other models as X and the difference between the main model and true target value as Y for every group with the option to choose between models. This chart is created using two parameters: `main_diff_model` (if None, the `model` parameter is used) and `compare_diff_models` (if None, the `models_to_compare` is used).

Group creation parameters:
- `d_groups_type` (str): Supported values are: `cut` - bin values into discrete intervals, `qcut` - quantile-based discretization function, `freq` - bins created using start, end and the length of each interval. 
- `d_bins` (int): Number of bins for `cut` and `qcut` groups_type. Default value is 10.
- `d_start` (float): Start value for `freq` groups_type. If not set, min(column)-1 is used.
- `d_end` (float): End value for `freq` groups_type. If not set, max(column) is used. 
- `d_freq` (float): The length of each interval for `freq` groups_type. Default value is 1.5.

**Features comparison chart**
A features comparison chartshows the predicted values of the compared models for each feature group with the option to choose between features.

Group creation parameters:
- `f_groups_type` (str, dict): Supported values are: `cut` - bin values into discrete intervals, `qcut` - quantile-based discretization function, `freq` - bins created using start, end and the length of each interval. If str, all features are cut using `f_groups_type`. If dict, must be {'feature': 'groups_type', 'all': 'groups_type'} where 'all' will be used for all features not listed in the dict.
- `f_bins` (int, dict): Number of bins for `cut` and `qcut` groups_type. If int, all features are cut using `f_bins`. If dict, must be {'feature': bins, 'all': 'groups_type'} where 'all' will be used for all features not listed in the dict. Default value is 10.
- `f_start` (float, dict): Start value for `freq` groups_type. If notset, min(column)-1 is used. If float, all features are cut using `f_start`. If dict, must be {'feature': start, 'all': 'groups_type'} where 'all' will be used for all features not listed in the dict.
- `f_end` (float, dict): End value for `freq` groups_type. If notset, max(column) is used. If float, all features are cut using `f_end`. If dict, must be {'feature': end, 'all': 'groups_type'} where 'all' will be used for all features not listed in the dict.
- `f_freq` (float, dict): The length of each interval for `freq` groups_type. Default value is 1.5. If float, all features are cut using `f_freq`. If dict, must be {'feature': freq, 'all': 'groups_type'} where 'all' will be used for all features not listed in the dict.

**Comparison matrix**
A comparison matrix is created for a pair of models initialized in the `pairs_for_matrix` parameter. 

Algorithm:
1. it creats dataframe with the first model predicted values, the second model predicted values and the target values;
2. groups are created using either `m_bins` or `m_freq`;
2. a matrix is created with these groups as columns and index;
3. while iterating over columns (as gr) and index (as gr_1), it finds values for the first model in gr AND for the second model in gr_2, gets rows with these values;
4. it counts the target values in these rows and divides them by their sum.

### Example
```python
r = Report(...,
           models_to_compare = [irf, igbm, iglm],
           comparison_metrics = [mean_squared_error, rmse],
           f_groups_type = {'all': 'cut', 'feat1': 'freq'}, f_bins = 15, f_freq = 10000,
           p_groups_type = 'cut', p_bins = 20,
           d_groups_type = 'cut', d_bins = 15,
           main_diff_model = igbm, compare_diff_models = [irf, irf_2, iglm],
           pairs_for_matrix = [[igbm, iglm],[igbm, irf]], m_bins = 20) 
```