# Tutorials
In this module different main features and functionalities of the `Insolver` library are introduced in the tutorials. 
These tutorials contain a demonstration of the use of various classes and methods. All datasets presented in the examples can be downloaded using the `download_dataset` function presented [here](https://github.com/MindSetLib/Insolver/blob/master/insolver/model_tools/model_utils.py).

## Сontents
- [Binary Classification Example - Rain in Australia Prediction](https://github.com/MindSetLib/Insolver/blob/master/tutorials/Binary%20Classification%20Example%20-%20Rain%20in%20Australia%20Prediction.ipynb)
This tutorial demonstrates how to create **classification models** for the [`weatherAUS`](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package) dataset: getting and preprocessing data, transformations, creating models, plotting SHAP values and comparing models.

- [Data Preprocessing Example I - New York City Airbnb](https://github.com/MindSetLib/Insolver/blob/master/tutorials/Data%20Preprocessing%20Example%20I%20-%20New%20York%20City%20Airbnb.ipynb)
This tutorial demonstrates how to use the [`feature_engineering`](https://github.com/MindSetLib/Insolver/tree/master/insolver/feature_engineering) module and all the **main features of each class**. For this, the [`AB_NYC_2019`](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data) dataset is used. 

- [Data Preprocessing Example II - New York City Airbnb](https://github.com/MindSetLib/Insolver/blob/master/tutorials/Data%20Preprocessing%20Example%20II%20-%20New%20York%20City%20Airbnb.ipynb)
This tutorial also demonstrates how to use the [`feature_engineering`](https://github.com/MindSetLib/Insolver/tree/master/insolver/feature_engineering) module, but it covers the **automated data preprossesing** class and all of its features. For this, the [`AB_NYC_2019`](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data) dataset is used. 

- [Gradient Boosting Example - Lending Club](https://github.com/MindSetLib/Insolver/blob/master/tutorials/Gradient%20Boosting%20Example%20-%20Lending%20Club.ipynb)
This tutorial demonstrates how to create **classification models** for the [`Lending Club`](https://www.kaggle.com/wordsforthewise/lending-club) dataset using the **Gradient Boosting libraries** and the `InsolverGBMWrapper` class.

- [Transforms Inference Example](https://github.com/MindSetLib/Insolver/blob/master/tutorials/Transforms%20Inference%20Example.ipynb)
This tutorial demonstrates how to load `InsolverTransform` transforms from a file using the `load_transforms` function.

- [InsolverDataFrame and InsolverTransform Example](https://github.com/MindSetLib/Insolver/blob/master/tutorials/InsolverDataFrame%20and%20InsolverTransform%20Example.ipynb)
This tutorial demonstrates main features of the `InsolverDataFrame` class and the `InsolverTransform` class.

- [Regression Example - FreeMLP](https://github.com/MindSetLib/Insolver/blob/master/tutorials/Regression%20Example%20-%20FreeMLP.ipynb)
This tutorial demonstrates how to create **regression models** for the `freMPL-R` dataset: getting and preprocessing data, transformations, creating models, plotting SHAP values and comparing models.

- [Regression Example - US Accidents](https://github.com/MindSetLib/Insolver/blob/master/tutorials/Regression%20Example%20-%20FreeMLP.ipynb)
This tutorial demonstrates how to create **regression models** for the [`US Traffic Accident`](https://smoosavi.org/datasets/us_accidents) dataset: getting and preprocessing data, transformations, creating models, plotting SHAP values and comparing models.

- [Report Example](https://github.com/MindSetLib/Insolver/blob/master/tutorials/Report%20Example.ipynb)
This tutorial demonstrates how to create a **HTML report** with different models using the `Report` class.
