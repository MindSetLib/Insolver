# Model Wrappers

Model wrappers allow you to perform some model fitting routines
using a unified interface across different backend packages.  Currently model wrappers implement
Generalized Linear Models (`scikit-learn`, `h2o`) and Gradient Boosting Machines (`xgboost`, `lightgbm`, `catboost`).

## BaseWrapper
`InsolverBaseWrapper` is an abstraction that unifies basic functionality of the wrapper which is not dependent
on the model type and backend choice. This class represents a building block for creating wrappers for specific models
with their backend packages.

By itself, `InsolverBaseWrapper` provides basic `__call__`, `load_model`, `save_model` methods. By default, `__call__`
method returns the model object itself which is stored in `self.model` attribute.

### Creating a new wrapper using `InsolverBaseWrapper`
A new wrapper class should have `InsolverBaseWrapper` as a parent class, which requires `backend` argument.
This argument should contain a string with the name of the backend which will be used in model fitting.
It should be set at class instance initialization, the usage example is provided below.

```python
from insolver.wrappers.base import InsolverBaseWrapper

class CustomWrapper(InsolverBaseWrapper):
    def __init__(self, backend, ...):
        super(CustomWrapper, self).__init__(backend)
        ...
```

Apart from that, a new wrapper class should also possess the following list of attributes: 
* `self.algo`: A string containing the name of the algorithm/model type the wrapper implement.
  
  Example: `self.algo = 'glm'`.
* `self._backends`: A list containing all implemented backends for the given algorithm/model type. Must be coherent with 
  the values passed in `backend`.
  
  Example: `self._backends = ['h2o', 'sklearn']`.
* `self._back_load_dict`: A dictionary containing backend names from `self._backends` as keys and callable functions as 
values. These functions should implement the process of loading the model from file to the wrapper when using
  `load_model` method. `InsolverBaseWrapper` implements function `_pickle_load` out of the box. 
  
  Example: `self._back_load_dict = {'backend1': self._pickle_load, 'backend2': self._pickle_load}`.
  
  **NB:** An extension for H2O framework, `InsolverH2OExtension` class, also implements function `_h2o_load` by default,
  but it should be used a bit differently.

* `self._back_save_dict`: A dictionary containing backend names from `self._backends` as keys and callable functions as 
values. These functions should implement the process of saving the model from the wrapper to file when using
  `save_model` method. `InsolverBaseWrapper` implements function `_pickle_save` out of the box. An extension for H2O
  framework, `InsolverH2OExtension` class, also implements function `_h2o_save` by default.
  
  Example: `self._back_load_dict = {'backend1': self._pickle_save, 'backend2': self._pickle_save}`.

* `self.object`: A callable function returning a new model object with parameters specified at class instance
  initialization of the wrapper. This function should take keyword arguments for other parameters of the model object 
  that were not defined at class instance initialization. The idea behind this object is somehow similar to the
  following. Assume you want to use `GridSearchCV` from `sklearn.model_selection` using `SVC()` from `sklearn.svm` with
  specific kernel type. Then you likely to create some object, say, `clf = SVC(kernel='rbf')` and then pass it to 
  `GridSearchCV(clf, ...)`. You may do whatever you want with `GridSearchCV`, but `clf` will not change due to this
  operations. Analogously, `self.object` protects initial model object from being changed. An example of usage
  `self.object` you can find below under `self.model` attribute.
  
* `self.model`: A model object, probably initialized using `self.object` attribute function. This attribute serves as a 
  placeholder for a resulting model. This attribute is used for most of the operations with wrappers, including calls.

  Example:
  ```python
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LinearRegression
  from insolver.wrappers.base import InsolverBaseWrapper

  class CustomWrapper(InsolverBaseWrapper):
      def __init__(self, backend, ..., kwargs):
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
  
The functionality of a `InsolverBaseWrapper` is quite limited. However, it allows adding extensions to the wrappers 
for performing routines such as cross-validation and hyperparameter optimization.  

## TrivialWrapper

## Generalized Linear Models

## Gradient Boosting Machines

## InsolverH2OExtension
`'h2o': partial(self._h2o_load, h2o_init_params=h2o_init_params)}`
