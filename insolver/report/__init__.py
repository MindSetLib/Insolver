try:
    from .report import Report
except ImportError as e:
    raise ImportError(f"Missing dependencies. Please install them with `pip install insolver[report]`. {e}")
