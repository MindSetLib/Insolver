class InsolverMain:

    def get_batch(self):
        """Gets data as InsolverDataFrame."""
        raise NotImplementedError("Class should implement 'get_batch'.")

    def get_meta_info(self):
        """Gets JSON with Insolver meta information."""
        raise NotImplementedError("Class should implement 'get_meta_json'.")


class InsolverTransformMain:
    def __init__(self):
        if not hasattr(self, 'priority'):
            raise NotImplementedError("Transformation class should have the 'priority' property.")