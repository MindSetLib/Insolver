# We can set up user transformations
class TransformSocioCateg:
    def __init__(self, column_socio_categ):
        self.priority = 0
        self.column_socio_categ = column_socio_categ

    def __call__(self, df):
        df[self.column_socio_categ] = df[self.column_socio_categ].str.slice(0,4)
        return df