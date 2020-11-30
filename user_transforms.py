import pandas as pd

from insolver.transforms.InsolverTransforms import InsolverTransformMain


class TransformSocioCateg(InsolverTransformMain):
    def __init__(self, column_socio_categ):
        self.priority = 0
        super().__init__()
        self.column_socio_categ = column_socio_categ

    def __call__(self, df):
        df[self.column_socio_categ] = df[self.column_socio_categ].str.slice(0,4)
        return df


class TransformToNumeric(InsolverTransformMain):
    def __init__(self, column_param, downcast='integer'):
        self.priority = 0
        super().__init__()
        self.column_param = column_param
        self.downcast = downcast

    def __call__(self, df):
        df[self.column_param] = pd.to_numeric(df[self.column_param], downcast=self.downcast)
        return df
