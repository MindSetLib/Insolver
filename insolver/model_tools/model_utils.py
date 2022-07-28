from sklearn.model_selection import train_test_split


def train_val_test_split(*arrays, val_size, test_size, random_state=0, shuffle=True, stratify=None):
    """Function for splitting dataset into train/validation/test partitions.

    Args:
        *arrays (array_like): Arrays to split into train/validation/test sets containing predictors.
        val_size (float): The proportion of the dataset to include in validation partition.
        test_size (float): The proportion of the dataset to include in test partition.
        random_state (int, optional): Random state, passed to train_test_split() from scikit-learn. (default=0).
        shuffle (bool, optional): Passed to train_test_split() from scikit-learn. (default=True).
        stratify (array_like, optional): Passed to train_test_split() from scikit-learn. (default=None).

    Returns:
        list: [x_train, x_valid, x_test, y_train, y_valid, y_test].

        A list of partitions of the initial dataset.
    """
    n_arrays = len(arrays)
    split1 = train_test_split(
        *arrays, random_state=random_state, shuffle=shuffle, test_size=test_size, stratify=stratify
    )
    if n_arrays > 1:
        train, test = split1[0::2], split1[1::2]
        if val_size != 0:
            split2 = train_test_split(
                *train,
                random_state=random_state,
                shuffle=shuffle,
                test_size=val_size / (1 - test_size),
                stratify=stratify
            )
            train, valid = split2[0::2], split2[1::2]
            return [*train, *valid, *test]
        else:
            return [*train, *test]
    else:
        train, test = split1[0], split1[1]
        if val_size != 0:
            split2 = train_test_split(
                train,
                random_state=random_state,
                shuffle=shuffle,
                test_size=val_size / (1 - test_size),
                stratify=stratify,
            )
            train, valid = split2[0], split2[1]
            return [train, valid, test]
        else:
            return [train, test]


def train_test_column_split(x, y, df_column):
    """Function for splitting dataset into train/test partitions w.r.t. a column (pd.Series).

    Args:
        x (pd.DataFrame): DataFrame containing predictors.
        y (pd.DataFrame): DataFrame containing target variable.
        df_column (pd.Series): Series for train/test split, assuming it is contained in x.

    Returns:
        tuple: (x_train, x_test, y_train, y_test).

        A tuple of partitions of the initial dataset.
    """
    x1, y1, col_name = x.copy(), y.copy(), df_column.name
    y1[col_name] = df_column
    return (
        x1[x1[col_name] == 'train'].drop(col_name, axis=1),
        x1[x1[col_name] == 'test'].drop(col_name, axis=1),
        y1[y1[col_name] == 'train'].drop(col_name, axis=1),
        y1[y1[col_name] == 'test'].drop(col_name, axis=1),
    )
