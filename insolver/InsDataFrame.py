import pandas as pd


class InsDataFrame:

    def load_pd(self, pd_dataframe):
        """
        Loads data from Pandas Dataframe.
        :param pd_dataframe: Pandas Dataframe.
        :return: None.
        """
        self._df = pd_dataframe

    def columns_match(self, match_from_to):
        """
        Matches columns in dataframe.
        :param match_from_to: Matching dict.
        :return: None.
        """
        self._df.rename(columns=match_from_to, inplace=True)

    # ---------------------------------------------------
    # Person data methods
    # ---------------------------------------------------

    _gender_dict = {'Male': 0, 'Female': 1}

    def transform_gender(self):
        """
        Transforms values in column 'Gender' from {'Male','Female'} to {0,1}.
        :return: None.
        """
        self._df['Gender'] = self._df['Gender'].map(self._gender_dict)

    @staticmethod
    def _age(age, age_max):
        if pd.isnull(age):
            age = None
        elif age < 18:
            age = None
        elif age > age_max:
            age = age_max
        return age

    def transform_age(self, age_max=70):
        """
        Transforms values of drivers' minimum age in column 'driver_minage' with values over 'age_max' grouped.
        :param age_max: Maximum value of drivers' age, bigger values will be grouped (70 by default).
        :return:
        """
        self._df['driver_minage'] = self._df['driver_minage'].apply(self._age, args=(age_max,))

    @staticmethod
    def _age_gender(age_gender):
        _age = age_gender[0]
        _gender = age_gender[1]
        if _gender == 0:  # Male
            _driver_minage_m = _age
            _driver_minage_f = 18
        elif _gender == 1:  # Female
            _driver_minage_m = 18
            _driver_minage_f = _age
        else:
            _driver_minage_m = 18
            _driver_minage_f = 18
        return [_driver_minage_m, _driver_minage_f]

    def transform_age_gender(self):
        """
        Gets intersections of drivers' minimum age and gender in columns 'driver_minage_m' and 'driver_minage_f' from
        columns 'driver_minage' and 'Gender'.
        :return: None.
        """
        self._df['driver_minage_m'], self._df['driver_minage_f'] = zip(
            *self._df[['driver_minage', 'Gender']].apply(self._age_gender, axis=1).to_frame()[0])

    @staticmethod
    def _exp(exp, exp_max):
        if pd.isnull(exp):
            exp = None
        elif exp < 0:
            exp = None
        elif exp > exp_max:
            exp = exp_max
        return exp

    def transform_exp(self, exp_max=52):
        """
        Transforms values of drivers' minimum experience in column 'driver_minexp' with values over 'exp_max' grouped.
        :param exp_max: Maximum value of drivers' experience, bigger values will be grouped (52 by default).
        :return: None.
        """
        self._df['driver_minexp'] = self._df['driver_minexp'].apply(self._exp, args=(exp_max,))

    # ---------------------------------------------------
    # Other data methods
    # ---------------------------------------------------

    def polynomizer(self, column, n=2):
        """
        Gets polynomial of feature.
        :param column: Feature's column name.
        :param n: Polinomial's degree.
        :return: None.
        """
        if column in list(self._df.columns):
            for i in range(2, n + 1):
                self._df[column + '_' + str(i)] = self._df[column] ** i

    def get_dummies(self, columns):
        """
        Gets dummy columns of the feature.
        :param columns: List of columns to transforme.
        :return: None.
        """
        self._df = pd.get_dummies(self._df, columns=columns)

    # ---------------------------------------------------
    # General methods
    # ---------------------------------------------------

    def info(self):
        return self._df.info()

    def head(self, columns, n=5):
        return self._df.head(n)

    def len(self):
        return len(self._df)

    def get_pd(self, columns):
        return self._df[columns].copy()
