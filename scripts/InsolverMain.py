class InsolverMain:

    def get_data(self):
        """
        Gets data as InsolverDataFrame.
        """
        raise NotImplementedError("Class should implement 'get_data'.")

    def get_meta_info(self):
        """
        Gets JSON with Insolver meta information.
        """
        raise NotImplementedError("Class should implement 'get_meta_json'.")
