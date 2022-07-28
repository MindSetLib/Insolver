def error_handler(return_footer):
    def inner_decorator(func):
        def inner_function(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f'Exception {e.__class__.__name__} in {func.__name__}: {e}')
                return [] if not return_footer else ({'footer': []}, [])

        return inner_function

    return inner_decorator
