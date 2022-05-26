import numpy as np
from scipy.special import xlogy


def deviance_score(y, y_pred, weight=None, power=0, agg='sum'):
    """Function for Deviance evaluation.

    Args:
        y: Array with target variable.
        y_pred: Array with predictions.
        weight: Weights for weighted metric.
        power:
        agg: Function to calculate deviance ['sum', 'mean'] or callable are supported.

    Returns:
        float, value of the Poisson deviance.
    """
    dict_func = {'sum': np.sum, 'mean': np.mean}
    func = dict_func[agg] if agg in ['sum', 'mean'] else agg if callable(agg) else None
    if func is None:
        raise ValueError
    weight = 1 if weight is None else weight
    if str(power).lower() in ["normal", "gaussian", "0"]:
        return func(weight * np.power(y - y_pred, 2))
    elif str(power).lower() in ["poisson", "1"]:
        return func(2 * weight * (xlogy(y, y / y_pred) - (y - y_pred)))
    elif str(power).lower() in ["gamma", "2"]:
        return func(2 * weight * (np.log(y_pred / y) + y / y_pred - 1))
    elif isinstance(power, str) or (0 < power < 1):
        raise Exception(f"power={power} is not supported.")
    else:
        return func(2 * weight * (np.power(np.max(y, 0), 2 - power) / ((1 - power) * (2 - power)) -
                                  (y * np.power(y_pred, 1 - power)) / (1 - power) +
                                  (np.power(y_pred, 2 - power)) / (2 - power)))


def deviance_poisson(y, y_pred, weight=None, agg='sum'):
    """Function for Poisson Deviance evaluation.

    Args:
        y: Array with target variable.
        y_pred: Array with predictions.
        weight: Weights for weighted metric.
        agg: Function to calculate deviance ['sum', 'mean'] or callable are supported.

    Returns:
        float, value of the Poisson deviance.
    """
    return deviance_score(y, y_pred, weight=weight, power=1, agg=agg)


def deviance_gamma(y, y_pred, weight=None, agg='sum'):
    """Function for Gamma Deviance evaluation.

    Args:
        y: Array with target variable.
        y_pred: Array with predictions.
        weight: Weights for weighted metric.
        agg: Function to calculate deviance ['sum', 'mean'] or callable are supported.

    Returns:
        float, value of the Gamma deviance.
    """
    return deviance_score(y, y_pred, weight=weight, power=2, agg=agg)


def deviance_explained(y, y_pred, weight=None, power=0):
    """Function for Pseudo R^2 (Deviance explained) evaluation.

        Args:
            y: Array with target variable.
            y_pred: Array with predictions.
            weight: Weights for weighted metric.
            power: Power for deviance calculation.

        Returns:
            float, value of the Pseudo R^2.
        """
    dev = deviance_score(y, y_pred, weight=weight, power=power)
    dev0 = deviance_score(y, np.repeat(np.mean(y), len(y)), weight=weight, power=power)
    return 1 - dev/dev0


def deviance_explained_poisson(y, y_pred, weight=None):
    """Function for Pseudo R^2 (Deviance explained) evaluation for Poisson model.

        Args:
            y: Array with target variable.
            y_pred: Array with predictions.
            weight: Weights for weighted metric.

        Returns:
            float, value of the Pseudo R^2.
        """
    return deviance_explained(y, y_pred, weight=weight, power=1)


def deviance_explained_gamma(y, y_pred, weight=None):
    """Function for Pseudo R^2 (Deviance explained) evaluation for Gamma model.

        Args:
            y: Array with target variable.
            y_pred: Array with predictions.
            weight: Weights for weighted metric.

        Returns:
            float, value of the Pseudo R^2.
        """
    return deviance_explained(y, y_pred, weight=weight, power=2)
