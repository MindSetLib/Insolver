from pandas import DataFrame
from insolver.transforms import InsolverTransform, import_transforms


class TransformExp:
    pass


class TransformToNumeric:
    pass


def test_import_transforms():
    ut = import_transforms(module_path='./examples/user_transforms.py')
    keys = set(ut.keys())
    assert all([x in keys for x in ['TransformExp', 'TransformSocioCateg', 'TransformToNumeric']])

    globals().update(ut)
    df = DataFrame({'LicAge': [100, 356, 57, 24, 0], 'StrToNum': ['-2', '1000', '33', '0', '765432']})

    df = InsolverTransform(df, [TransformExp('LicAge', 57),
                                TransformToNumeric(column_names=['StrToNum'], downcast='integer')])
    df.ins_transform()
    assert DataFrame(df).compare(DataFrame({'LicAge': [57, 57, 57, 24, 0],
                                            'StrToNum': [-2, 1000, 33, 0, 765432]})).empty
