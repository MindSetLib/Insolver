try:
    from .plots import ExplanationPlot
    from .dice import DiCEExplanation
    from .lime import LimeExplanation
    from .shap import SHAPExplanation
except ImportError as e:
    raise ImportError(f"Missing dependencies. Please install them with `pip install insolver[interpretation]`. {e}")
