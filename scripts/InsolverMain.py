class InsolverMain:

    def get_pd(self):
        """
        Gets the Pandas DataFrame from the backend.
        """
        raise NotImplementedError("Class should implement 'get_pd'.")

    def get_meta_json(self):
        """
        Gets the JSON with meta information the backend.
        """
        raise NotImplementedError("Class should implement 'get_meta_json'.")
