import pandas as pd
from dice_ml import Dice, Data, Model
from .base import InterpretBase


class DiCEExplanation(InterpretBase):
    """
    Diverse Counterfactual Explanations (DiCE). Uses dice_ml package. Supports only sklearn models.

    Parameters:
        estimator: Trained sklearn model.
        model_type (str): Type of the model. Values `classifier` and `regressor` are supported.
        x (pandas.Dataframe, optional): X value or dataset. If dataset, outcome_name is required.
        y: Y value.
        continuous_features (list): Continuous features list.
        outcome_name (str): Outcome feature name.
        method (str): Name of the method to use for generating counterfactuals. Values `genetic`, `random` and `kdtree`
         are supported.

    """

    def __init__(self, estimator, model_type, x, y=None, continuous_features=None, outcome_name=None, method='genetic'):
        self.estimator = estimator
        self.model_type = model_type
        self.x = x
        self.y = y
        self.continuous_features = [] if continuous_features is None else continuous_features
        self.outcome_name = outcome_name
        self.method = method

        # raise error if y or outcome_name are not initialized
        if self.y is None and self.outcome_name is None:
            raise AttributeError('Either y or outcome_name must be initialized.')

    def show_explanation(
        self,
        instance,
        total_CFs=2,
        desired_range=None,
        desired_class='opposite',
        features_to_vary='all',
        permitted_range=None,
        result_type='dataframe',
        return_json=False,
    ):
        """
        Show explanation. Creates dice_ml.Data, dice_ml.Model, dice_ml.Dice using Data and Model, then generates
        counterfactuals using dice_ml.Dice.generate_counterfactuals() method.

        Parameters:
            instance: Instance to use to generate counterfactuals.
            total_CFs (int): Number of counterfactuals.
            desired_range (list): Desired range of the outcome values if the model is a regression.
            desired_class (int): Desired class of the outcome values if the model is a classification. Can take 0 or 1.
            features_to_vary: Either a string “all” or a list of feature names to vary.
            permitted_range (dict): Dictionary with feature names as keys and permitted range in list as values.
            result_type (str): Type of the result to visualize. Values `dataframe` and `list` are supported.
            return_json (bool): Flag whether to return json.

        Raises:
            NotImplementedError: If result type is not supported.

        """
        dataframe = self.x if self.y is None else pd.concat([self.x, self.y], axis=1)
        _outcome_name = self.outcome_name if self.outcome_name else self.y.name

        # create dice_ml.Data class containing all required information about the data
        data = Data(dataframe=dataframe, continuous_features=list(self.continuous_features), outcome_name=_outcome_name)

        # create dice_ml.Model class that returns class selected with the backend parameter
        dice_model = Model(model=self.estimator, backend='sklearn', model_type=self.model_type)

        # create dice_ml.Dice class that returns class selected with the method parameter
        self.model = Dice(data, dice_model, method=self.method)

        # generate counterfactuals
        counterfactuals = self.model.generate_counterfactuals(
            query_instances=instance,
            total_CFs=total_CFs,
            desired_range=desired_range,
            desired_class=desired_class,
            features_to_vary=features_to_vary,
            permitted_range=permitted_range,
        )

        # visualize as dataframe
        if result_type == 'dataframe':
            counterfactuals.visualize_as_dataframe()

        # else visualize as list
        elif result_type == 'list':
            counterfactuals.visualize_as_list()

        # else raise error
        else:
            raise NotImplementedError(f'{result_type} is not supported.')

        return counterfactuals if not return_json else counterfactuals.to_json()

    def plot(self, figsize, **kwargs):
        super.plot()

    def get_model(self):
        """
        Get model.
        """
        return self.model
