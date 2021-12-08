# Model Wrappers

Model wrappers allow you to perform model fitting routines using a unified interface from different backend packages.
Currently, model wrappers implement Generalized Linear Models (`scikit-learn`, `h2o`) and Gradient Boosting Machines (`xgboost`, `lightgbm`, `catboost`).

## BaseWrapper
`InsolverBaseWrapper` is an abstract class that unifies the basic functionality of the wrapper, which is not dependent on the model type and backend choice. This class represents a building block for creating wrappers for specific models with their backend packages.

By itself, `InsolverBaseWrapper` provides basic `__call__`, `load_model`, `save_model` methods. By default, `__call__` method returns the model object itself which is stored in `self.model` attribute.

### Creating a new wrapper using `InsolverBaseWrapper`
A new wrapper class should have `InsolverBaseWrapper` as a parent class, which requires a `backend` argument. This argument should contain a string with the backend's name, which will be used in model fitting. It should be set at class instance initialization; the usage example is provided below.

```python
from insolver.wrappers.base import InsolverBaseWrapper

class CustomWrapper(InsolverBaseWrapper):
    def __init__(self, backend):
        super(CustomWrapper, self).__init__(backend)
        ...
```

Apart from that, a new wrapper class should also have the following list of attributes: 
* `self.algo`: A string containing the name of the algorithm/model type the wrapper implement.
  
  Example: `self.algo = 'glm'`.
* `self._backends`: A list containing all implemented backends for the given algorithm/model type. Must be coherent with the values passed in `backend`.
  
  Example: `self._backends = ['h2o', 'sklearn']`.
* `self._back_load_dict`: A dictionary containing backend names from `self._backends` as keys and callable functions as values. These functions should implement the process of loading the model from file to the wrapper when using `load_model` method. `InsolverBaseWrapper` implements function `_pickle_load` out of the box. 
  
  Example: `self._back_load_dict = {'backend1': self._pickle_load, 'backend2': self._pickle_load}`.
  
::::{important} An extension for H2O framework, `InsolverH2OExtension` class, also implements function `_h2o_load` by default, but it should be used a bit differently.
:::: 

* `self._back_save_dict`: A dictionary containing backend names from `self._backends` as keys and callable functions as values. These functions should implement the process of saving the model from the wrapper to the file using the `save_model` method. `InsolverBaseWrapper` implements function `_pickle_save` out of the box. An extension for H2O framework, `InsolverH2OExtension` class, also implements function `_h2o_save` by default.
  
  Example: `self._back_load_dict = {'backend1': self._pickle_save, 'backend2': self._pickle_save}`.

* `self.object`: A callable function returning a new model object with parameters specified at class instance initialization of the wrapper. This function should take keyword arguments for other parameters of the model object that were not defined at class instance initialization. The idea behind this object is somehow similar to the following. Assume you want to use `GridSearchCV` from `sklearn.model_selection` using `SVC()` from `sklearn.svm` with specific kernel type. Then you are likely to create some object, say, `clf = SVC(kernel='rbf')` and then pass it to `GridSearchCV(clf, ...)`. You may do whatever you want with `GridSearchCV`, but `clf` will not change due to these operations. Analogously, `self.object` protects the initial model object from being changed. You can find an example of usage `self.object` below the `self.model` attribute.
  
* `self.model`: A model object, probably initialized using `self.object` attribute function. This attribute serves as a placeholder for a resulting model. This attribute is used for most operations with wrappers, including calls.

  Example:
  ```python
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LinearRegression
  from insolver.wrappers.base import InsolverBaseWrapper

  class CustomWrapper(InsolverBaseWrapper):
      def __init__(self, backend, kwargs):
          super(CustomWrapper, self).__init__(backend)
          self.algo, self._backends = 'custom', ['sklearn']
          self._back_load_dict = {'sklearn': self._pickle_load}
          self._back_save_dict = {'sklearn': self._pickle_save} 
          self.params = kwargs
        
          def __params_pipe(**parameters):
              parameters.update(self.params)
              return Pipeline([('scaler', StandardScaler()),
                               ('reg', LinearRegression(**parameters))])
          self.model, self.object = __params_pipe(**self.params), __params_pipe 
  ```
  
The functionality of an `InsolverBaseWrapper` is quite limited. However, it allows adding extensions to the wrappers for performing cross-validation and hyperparameter optimization.  

## TrivialWrapper

```{eval-rst}
.. autoclass:: insolver.wrappers.InsolverTrivialWrapper
    :show-inheritance:
```


Although `InsolverTrivialWrapper` does not provide an actual model, it may be a useful benchmark in model comparison. 

`InsolverTrivialWrapper` requires two optional arguments: `col_name` and `agg`. If the `col_name` argument is absent, fitted `InsolverTrivialWrapper` make "predictions" by returning the value of applied `agg` callable function on the training data. If the `agg` argument is not specified, `np.mean` is used. The `col_name` argument takes strings or a list of strings as values. This object makes sense mainly in the case of the regression problem. In the classification problem, it is not very useful since such an object does not support predicting labels.

The resulting "predictions" are obtained as follows:
1. On the `fit` step, groupby operation (w.r.t. columns in `col_name`) with `agg` aggregation function is applied to the target values.
2. On the `predict` step, the first step results are mapped to the values in `X` of the `predict` method using the `col_name` argument as a key. In cases when there are no matches with `col_name` values in training data, the value of `agg` is used taken over the whole training set.

## Generalized Linear Models

```{eval-rst}
.. autoclass:: insolver.wrappers.InsolverGLMWrapper
    :show-inheritance:
```

`InsolverGLMWrapper` implements Generalized Linear Models with the support of `h2o` and `scikit-learn` packages.

`InsolverGLMWrapper` supports methods `coef()` and `norm_coef()` that output all the model coefficients. Although the models are fitted using a standardized dataset, the coefficients in `coef()` are recalculated so that the prediction may be obtained as an inverse link function applied to a linear combination of coefficients and factors in `X`. 

### GLM using `sklearn` backend
Insolver uses the functionality of [`TweedieRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html) class from [scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-regression). However, insolver fits a pipeline consisting of two steps, the first step with `StandardScaler` and the second with `TweedieRegressor` itself. By default, `StandardScaler` is used with `with_mean` and `with_std` arguments equal to `True` since the optimization procedure for non-standardized data may fail. Also, the insolver makes available string names of the distributions for the `family` parameter; the available options are: `gaussian` or `normal`, `poisson`, `gamma` and `inverse_gaussian`. This parameter can also accept a numeric value for Tweedie power if `family` is not in the range (0, 1).

GLM with `sklearn` backend also supports hyperparameter optimization using `hyperopt_cv()` method.

### GLM using `h2o` backend
Insolver uses the functionality of [`H2OGeneralizedLinearEstimator`](http://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/modeling.html#h2ogeneralizedlinearestimator) class from [H2O](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/glm.html).

`InsolverGLMWrapper` with `h2o` backend supports training data on the train and validation sets with a sample weight parameter via `offset_column` from `h2o`.

GLM with `h2o` backend also supports hyperparameter optimization using `optimize_hyperparam()` method.

## Gradient Boosting Machines

```{eval-rst}
.. autoclass:: insolver.wrappers.InsolverGBMWrapper
    :show-inheritance:
```

`InsolverGBMWrapper` implements Gradient Boosting Machines with support of `xgboost`, `lightgbm` and `catboost` packages. This object supports only classification and regression problems; that is why objective functions for ranking may not work well.

Gradient boosting wrapper can also interpret feature importance of the fitted models on the given dataset via `shap` using `shap()` method and analyze factor contributions with waterfall charts using `shap_explain()`.

It is also possible to examine metrics and SHAP values changes on cross-validation folds with `cross_val()`.

`InsolverGLMWrapper` with all three backends supports hyperparameter optimization using the `hyperopt_cv()` method.

## Random Forest

```{eval-rst}
.. autoclass:: insolver.wrappers.InsolverRFWrapper
    :show-inheritance:
```

`InsolverRFWrapper` implements Random Forest Models with `scikit-learn` support. It uses [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) to create a classification model and [`RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor) to create a regression model. You can change model type by setting the `task` attribute to `class` or `reg`, respectively. 

Class `InsolverRFWrapper` can also perfom cross-validation using method `cross_val`.