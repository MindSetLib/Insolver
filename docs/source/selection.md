#Selection

## Feature Selection

```{eval-rst}
 .. autoclass:: insolver.selection.FeatureSelection
    :show-inheritance: 
```
Class `FeatureSelection` allows you to compute features importances using selected method. It also can plot it with selected plot size and the importance threshold. Using computed importance you can create a new dataset with the best features. You can also use permutation importance model inspection technique with some models.

Class `FeatureSelection` supports such tasks as classification, regression, multiclass classification and multiclass multioutput classification.

The following  methods can be used for each individual task:
- for the `class` task Mutual information, F statistics, chi-squared test, Random Forest, Lasso or ElasticNet can be used;
- for the `reg` task Mutual information, F statistics, Random Forest, Lasso or ElasticNet can be used;
- for the `multiclass` task Random Forest, Lasso or ElasticNet can be used;
- for the `multiclass_multioutput` classification Random Forest can be used.

Random Forest is used by default.

All the methods used in this class are from `scikit-learn`:  
- `random_forest`[classification model](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) / [regression model](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html), 
- `lasso` [classification model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) / [regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html),
-  `elasticnet` [classification model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) / [regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html),
-  `mutual_inf` [classification information](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html) / [regression information](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html),
-  `f_statistic` [classification statistic](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html) / [regression statistic](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html),
-  `chi2` [classification statistic](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html).

[`Permutation feature importance`](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html) technique is also from `scikit-learn`. It supports only [estimator](https://scikit-learn.org/stable/glossary.html#term-estimator) models: Random Forest, Lasso and ElasticNet.

## Methods diagram

![feature selection methods](feature_selection_methods.png)

## Example

```python
import pandas as pd
from insolver.frame import InsolverDataFrame
from insolver.selection import FeatureSelection

#create dataset using InsolverDataFrame or pandas.DataFrame
dataset = InsolverDataFrame(pd.read_csv("..."))

#init class FeatureSelection with default method
fs = FeatureSelection(y_column='y_column', task='class')

#create model using create_model()
fs.create_model(dataset)

#plot created model importances using plot_importance()
fs.plot_importance()

#create permutation importance using create_permutation_importance()
fs.create_permutation_importance()

#create new dataset using create_new_dataset()
new_dataset = fs.create_new_dataset()

#you can also create permutation importance by setting parameter permutation_importance=True
fs_p = FeatureSelection(method='lasso', task='class', permutation_importance=True)
fs_p.create_model(dataset)
```

## Sampling

[Sampling](https://en.wikipedia.org/wiki/Sampling_(statistics)) is the selection of a subset (a statistical sample) of individuals from within a statistical population to estimate characteristics of the whole population.
Class `Sampling` implements methods from __probability sampling__. A probability sample is a sample in which every unit in the population has a chance (greater than zero) of being selected in the sample, and this probability can be accurately determined.
There are four methods you can use by changing `method` parameter:
- `simple` (default) sampling is a technique in which a subset is randomly selected number from a set;
- `systematic` sampling is a technique in which a subset is selected from a set using a defined step;
- `cluster` sampling is a technique in which a set is divided into clusters, then the set is determined by a randomly selected number of clusters; 
- `stratified` sampling is a technique in which a set is divided into clusters, then the set is determined by a randomly selected number of units from each cluster.

The `n` parameter is used differently in each sampling method:
- for a `simple` sampling `n` is the number of values to keep;
- for a `systematic` sampling `n` is the number of step size;
- for a `cluster` sampling `n` is the number of clusters to keep;
- for a `stratified` sampling `n` is the number of values to keep in each cluster.

You can use dataframe column as clusters by defining `cluster_column`. It will use values from this column in `cluster` and `stratified` methods.

### Example

```python
import pandas as pd
from insolver import InsolverDataFrame, FeatureSelection

#create dataset using InsolverDataFrame or pandas.DataFrame
dataset = InsolverDataFrame(pd.read_csv("..."))

#create class instance with the selected sampling method
sampling = Sampling(n=10, n_clusters=5, method='stratified')

#use method sample_dataset() to create new dataframe
new_dataset = sampling.sample_dataset(df=dataset)

#using dataframe column as clusters
samling = Sampling(n = 2, cluster_column = 'name', method='stratified')
new_dataset = sampling.sample_dataset(df=dataset)
```

## Dimensionality Reduction

.. autoclass:: insolver.selection.DimensionalityReduction
    :show-inheritance:
    
`DimensionalityReduction` class allows you to reduce the dimensionality of data with a selected method. There are 3 types of techniques implemented: decomposition, manifold and discriminant analysis.

The type of the method can be specified in the `method` parameter. The list of methods that can be assigned is presented below. All methods are implemented from [scikit-learn](https://scikit-learn.org/stable/index.html).

Matrix decomposition is represented by methods such as:
- `pca` - [Principal Component Analysis, PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html);
- `svd` - [truncated Singular Value Decomposition, SVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html);
- `fa` - [Factor Analysis, FA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html);
- `nmf` - [Non-Negative Matrix Factorization, NMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html).

Discriminant Analysis is represented by methods such as:
- `lda` - [Linear Discriminant Analysis, LDA](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html).

Manifold learning is represented by methods such as:
- `lle` - [Locally Linear Embedding, LLE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html);
- `isomap` - [Isomap Embedding](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html);
- `t_sne` - [T-distributed Stochastic Neighbor Embedding, T-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).

Use `transform(X, y=None, **kwargs)` method to create a new transformed X. Parameters assigned as `kwargs` can be used to change model(estimator) parameters which can be found in the sklearn pages above.

You can plot the transformed X and y with the `plot_transformed(self, y, figsize=(10,10), **kwargs)` method. It uses [seaborn](https://seaborn.pydata.org/) to create plots. If the number of components is less than 3, [seaborn.scatterplot](https://seaborn.pydata.org/generated/seaborn.scatterplot.html) will be created, else [seaborn.pairplot](https://seaborn.pydata.org/generated/seaborn.pairplot.html) will be created. The `y` parameter is used as the hue. Parameters assigned as `kwargs` can be used to change plot parameters which can be found in the seaborn pages.

You can access created model with the `estimator` attribute.

### Example
```python
import pandas as pd
from insolver.selection import DimensionalityReduction

#create X and y
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = pd.DataFrame(iris.target, columns=['y']) 

#create DimensionalityReduction
dm = DimensionalityReduction(method='nmf')

#use transform() to create new X
new_X = dm.transform(X=X, n_components=3)

#plot result
dm.plot_transformed(y, figsize=(5,5), palette='Set2')
```