# Report
The report module allows you to create an HTML report containing information about your data and model.  

Report HTML file rendered by `jinja2` library, template of report file is based on bootstrap library template [link](https://getbootstrap.com/docs/5.1/examples/cheatsheet/).

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

In `insolver.report.Report` sections, articles, and parts are stored in the `sections` parameter of the Report class. `Article` is a `dict` containing the article's name, `list` of parts, header, and footer. Header and footer are optional fields where you can add comments before (header) and after (footer) content. Every `part` is a python string containing HTML code, for example `<table> ... </table>` for table.

`sections` list initialization from Report class:
```
self.sections = [
        {
          'name': 'Dataset',
          'articles': [
              {
                'name': 'Pandas profiling',
                'parts': ['<div class="col-12"><button ./profiling_report.html\';">Go to report</button></div>'],
                'header': '',
                'footer': '',
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
              ...
           ],
        },
     ]
```

Part of HTML template where section list is used:
```html
{% for section in sections %}
<section id="{{ section.name }}">
   <h2 class="sticky-xl-top fw-bold pt-3 pt-xl-5 pb-2 pb-xl-3">{{ section.name }}</h2>

   {% for article in section.articles %}
       <article class="my-3" id="{{ article.name }}">
     <div class="bd-heading sticky-xl-top align-self-start mt-5 mb-3 mt-xl-0 mb-xl-2">
       <h3>{{ article.name }}</h3>
       <a class="d-flex align-items-center"></a>
     </div>
     <div>
       <div class="bd-example">
        <p class="lead">{{ article.header }}</p>
       </div>
          <div class="bd-example">

             {% for item in article.parts %}
                     {{ item }}
             {% endfor %}

         </div>
       <p class="lead">{{ article.footer }}</p>
     </div>
   </article>
   {% endfor %}

 </section>
{% endfor %}

```

## Report content
The rendered report has two sections, **Data report** and **Model report**

### Data report
Data report is created using `pandas-profiling` [library](https://github.com/pandas-profiling/pandas-profiling).

To generate a profiling report, all data passed to a Report class is combined and sent to the `pandas_profiling.ProfileReport` class.
```python
from pandas_profiling import ProfileReport

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

```
The profiling report is saved as an HTML file in the report directory in the final stage.
```python
def to_html(self, path: str = '.', report_name: str = 'report'):
    ...
    self.profile.to_file(f"{path}/{report_name}/profiling_report.html")
    ...
```

### Model report
The model report describes feature importance coefficients, metrics, and model parameters.

#### Coefficients
Coefficients are a numerical representation of features impact on predictions of the model, calculated depending on model type:
- **Random Forest**: coefficients from `RandomForest.feature_importance_` parameter.
- **Linear Model**: estimated coefficients for the linear model taken from `InsolverGMLWrapper.coef()` method.
- **Boosting Models**: SHAP interaction values computed for a boosting model.

#### Metrics
Metrics are calculated depending on the type of task (`classification` or `regression`).

#### Parameters
Inner state of model instance (parameters).
