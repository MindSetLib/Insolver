try:
    from .dimensionality_reduction import DimensionalityReduction
    from .feature_selection import FeatureSelection
    from .sampling import Sampling
    from .smoothing import Smoothing
    from .normalization import Normalization
    from .feature_engineering import DataPreprocessing
except ImportError as e:
    raise ImportError(
        f"Missing dependencies. Please install them with `pip install insolver[feature_engineering]`. {e}"
    )
