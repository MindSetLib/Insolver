import warnings


def warn_insolver(msg, category, filter_='always'):
    def warning_format(message, category_, *args, **kwargs):
        return f"{category_.__name__}: {message}\n"

    defailt_format = warnings.formatwarning
    warnings.formatwarning = warning_format
    warnings.simplefilter(filter_, category)
    warnings.warn(msg, category)
    warnings.formatwarning = defailt_format
