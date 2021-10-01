import json
from pandas import DataFrame
from insolver.model_tools import train_val_test_split


class InsolverDataFrame(DataFrame):
    """Primary DataFrame class for Insolver.

    Attributes:
        df (:obj:`pd.DataFrame`): pandas DataFrame.
    """
    def __init__(self, df):
        super(InsolverDataFrame, self).__init__(df)
        if not isinstance(df, DataFrame):
            raise NotImplementedError("'df' should be the pandas DataFrame.")

    def get_batch(self):
        pass

    def get_meta_info(self):
        """Gets JSON with Insolver meta information.

        Returns:
            dict: Meta information JSON.
        """
        meta_json = {
            'type': 'InsolverDataFrame',
            'len': self.shape[0],
            'columns': []
        }
        for column in self.columns:
            meta_json['columns'].append({'name': column, 'dtype': self[column].dtypes, 'use': 'unknown'})
        return meta_json

    def split_frame(self, val_size, test_size, random_state=0, shuffle=True, stratify=None):
        """Function for splitting dataset into train/validation/test partitions.

        Args:
            val_size (float): The proportion of the dataset to include in validation partition.
            test_size (float): The proportion of the dataset to include in test partition.
            random_state (:obj:`int`, optional): Random state, passed to train_test_split() from scikit-learn
             (default=0).
            shuffle (:obj:`bool`, optional): Passed to train_test_split() from scikit-learn (default=True).
            stratify (:obj:`array_like`, optional): Passed to train_test_split() from scikit-learn (default=None).

        Returns:
            tuple: (train, valid, test). A tuple of partitions of the initial dataset.
        """
        return train_val_test_split(self, val_size=val_size, test_size=test_size, random_state=random_state,
                                    shuffle=shuffle, stratify=stratify)

    def sample_request(self, batch_size=1):
        """Create json request by a random sample from InsolverDataFrame

        Args:
            batch_size: number of random samples

        Returns:
            request (dict)
        """
        data_str = self.sample(batch_size).to_json()
        data = json.loads(data_str)
        request = {'df': data}
        return request
