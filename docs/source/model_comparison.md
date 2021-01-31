# Model comparison
Insolver has a small tool `ModelMetricsCompare` that allows to compare model metrics and prediction distributions for 
supervised learning algorithms.

`ModelMetricsCompare` calculates metrics and prediction statistics on a given dataset, that are calculated based on
insolver models given in `source` argument. Source may be either a list of models or a destination path to the folder 
with models.