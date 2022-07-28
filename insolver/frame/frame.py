import json
from typing import Type, Optional, List, Dict, Union, Any
from numpy import dtype as numpy_dtype
from pandas import DataFrame
from insolver.model_tools import train_val_test_split


class InsolverDataFrame(DataFrame):
    def __init__(
        self,
        data: Any = None,
        index: Any = None,
        columns: Any = None,
        dtype: Optional[numpy_dtype] = None,
        copy: Optional[bool] = None,
    ) -> None:
        """Primary DataFrame class for Insolver. Almost the same as the pandas.DataFrame.

        Args:
            data (ndarray (structured or homogeneous), Iterable, dict, or pandas.DataFrame): Dict can contain
             `pandas.Series`, arrays, constants, dataclass or list-like objects. If data is a dict, column order follows
             insertion-order. If a dict contains `pandas.Series` which have an index defined, it is aligned by its index
             (default=None).
            index (pandas.Index or array-like): Index to use for resulting frame. Will default to RangeIndex if no
             indexing information part of input data and no index provided.
            columns (pandas.Index or array-like): Column labels to use for resulting frame when data does not have them,
             defaulting to `pandas.RangeIndex(0, 1, 2, …, n)`. If data contains column labels, will perform column
             selection instead (default=None).
            dtype (numpy.dtype): Data type to force. Only a single dtype is allowed. If `None`, infer (default=None).
            copy (bool) Copy data from inputs. For dict data, the default of None behaves like `copy=True`. For
             `pandas.DataFrame` or 2d ndarray input, the default of `None` behaves like copy=False (default=None).

        """
        super(InsolverDataFrame, self).__init__(data, index, columns, dtype, copy)

    @property
    def _constructor(self) -> Type["InsolverDataFrame"]:
        return InsolverDataFrame

    def get_meta_info(self) -> Dict[str, Union[str, int, List[Dict[str, Union[str, numpy_dtype]]]]]:
        """Gets JSON with Insolver meta information.

        Returns:
            dict: Meta information JSON.
        """
        meta_json = {'type': 'InsolverDataFrame', 'len': self.shape[0], 'columns': list()}
        for column in self.columns:
            meta_json['columns'].append({'name': column, 'dtype': self[column].dtypes, 'use': 'unknown'})
        return meta_json

    def split_frame(
        self,
        val_size: float,
        test_size: float,
        random_state: Optional[int] = 0,
        shuffle: bool = True,
        stratify: Any = None,
    ) -> List[DataFrame]:
        """Function for splitting dataset into train/validation/test partitions.

        Args:
            val_size (float): The proportion of the dataset to include in validation partition.
            test_size (float): The proportion of the dataset to include in test partition.
            random_state (int, optional): Random state, passed to train_test_split() from scikit-learn
             (default=0).
            shuffle (bool, optional): Passed to train_test_split() from scikit-learn (default=True).
            stratify (array_like, optional): Passed to train_test_split() from scikit-learn (default=None).

        Returns:
            list: (train, valid, test). A list of partitions of the initial dataset.
        """
        return train_val_test_split(
            self, val_size=val_size, test_size=test_size, random_state=random_state, shuffle=shuffle, stratify=stratify
        )

    def sample_request(self, batch_size: int = 1) -> Dict[str, object]:
        """Create json request by a random sample from InsolverDataFrame

        Args:
            batch_size: number of random samples

        Returns:
            request (dict)
        """
        if batch_size == 1:
            data_str = self.sample(batch_size).iloc[0].to_json()
        else:
            data_str = self.sample(batch_size).to_json()
        data = json.loads(data_str)
        request = {'df': data}
        return request
