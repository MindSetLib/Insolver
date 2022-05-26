"""
Insolver settings for project.
"""

# Will ot work with formula from settings file?
FORMULA_CALCULATION = True

# Variables list from formula
VARIABLES_LIST = ['cf1', 'ac1', 'cf2', 'ac2', 'cf3', 'ac3', 'cf4', 'ac4', 'cf5', 'ac51', 'ac52', 'cf6', 'ac6']
# VARIABLES_LIST = ['cf1', 'cf2']

# models should be in models folder with _model postfix and transformation should be with _transform postfix:
# for example for models: cf1_model, ac1_model and for transformations cf1_transform, ac1_transform


# Final formula for calculation, variables should be from variables list
FORMULA = "cf1 * ac1 + cf2 * ac2 + cf3 * ac3 + cf4 * ac4 + cf5 * (ac51 + ac52) + cf6 * ac6"
# FORMULA = "cf1 + cf2"

# Number of cores for model inference
N_CORES = 10
