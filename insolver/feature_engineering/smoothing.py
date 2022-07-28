import matplotlib.pyplot as plt
from seaborn import scatterplot
from pandas import DataFrame
import statsmodels.api as sm
from scipy.signal import savgol_filter
from scipy.fft import rfft, rfftfreq, irfft


class Smoothing:
    """
    Smoothing algorithms.

    Parameters:
        method (str): Smoothing method. Values `moving_average`, `lowess`, `s_g_filter`, `fft` are supported.
        x_column (str): Name of the column to transform.
        y_column (str): Name of the y column for the `lowess` method.
        window (int): Window size for the `moving_average` and `s_g_filter` methods.
        polyorder (int): Polyorder for the `s_g_filter` method.
        threshold (float): Threshold for the `fft` method.

    Attributes:
        new_df (pandas.DataFrame): A new dataframe as a copy of the original data with a transformed column added.

    """

    def __init__(self, method, x_column=None, y_column=None, window=11, polyorder=5, threshold=1e5):
        self.method = method
        self.window = window
        self.polyorder = polyorder
        self.threshold = threshold
        self.x_column = x_column
        self.y_column = y_column
        self.new_df = DataFrame()

    def transform(self, data, **kwargs):
        """
        Main Smoothing method.
        It creates new `pandas.DataFrame` as a copy of original data and adds a new transformed column.

        Parameters:
            data (pandas.Dataframe, optional): Original data. If not `pandas.DataFrame`, new dataframe will be created.

        Raises:
            NotImplementedError: If method is not supported.
        """
        # initialize all methods
        self._init_methods_dict()

        # raise error if the method is not supported
        if self.method not in self.methods_dict.keys():
            raise NotImplementedError(f'Method {self.method} is not supported.')

        # if the data is a DataFrame create a copy
        if isinstance(data, DataFrame):
            self.new_df = data.copy()

        # else create a DataFrame from data with the column 'data'
        else:
            self.new_df = DataFrame(data, columns=['data'])
            self.x_column = 'data'

        # get a function and call it
        func = self.methods_dict[self.method]
        func(self.new_df, **kwargs)

        return self.new_df

    def _moving_average(self, df, **kwargs):
        """
        Moving Average.

        Parameters:
            df (pandas.Dataframe): New dataframe.
            **kwargs: Arguments for the `pandas.DataFrame.rolling` function.
        """
        # create a new column using the DataFrame.rolling().mean() methods
        df[f'{self.x_column}_Moving_Average'] = df[self.x_column].rolling(window=self.window, **kwargs).mean()

    def _lowess(self, df, **kwargs):
        """
        Locally Weighted Scatterplot Smoothing, LOWESS.

        Parameters:
            df (pandas.Dataframe): New dataframe.
            **kwargs: Arguments for the `statsmodels.api.nonparametric.lowess` function.
        """
        # create lowess from data
        lowess = sm.nonparametric.lowess(df[self.y_column], df[self.x_column], **kwargs)

        # save lowess shape for use in the plot_transformed() method
        self._lowess_shape = lowess.shape[1]

        # the returned from lowess array is two-dimensional if return_sorted is True
        if self._lowess_shape == 2:
            # add both of them to the DataFrame as new columns
            df[f'{self.x_column}_Lowess'] = lowess[:, 0]
            df[f'{self.y_column}_Lowess'] = lowess[:, 1]

        # the returned from lowess array is one dimensional if return_sorted is False
        else:
            df[f'{self.x_column}_Lowess'] = lowess

    def _savitzky_golaay(self, df, **kwargs):
        """
        Savitzky–Golay filter.

        Parameters:
            df (pandas.Dataframe): New dataframe.
            **kwargs: Arguments for the `scipy.signal.savgol_filter` function.
        """
        # create a new column with the savgol_filter method
        df[f'{self.x_column}_Savitzky_Golaay'] = savgol_filter(
            df[self.x_column], window_length=self.window, polyorder=self.polyorder, **kwargs
        )

    def _fft(self, df, **kwargs):
        """
        Fast Fourier Transform, FFT.

        Parameters:
            df (pandas.Dataframe): New dataframe.
        """
        # get signal from data
        signal = df[self.x_column]
        # compute the one-dimensional discrete Fourier Transform
        fourier = rfft(signal)
        # get the Discrete Fourier Transform sample frequencies
        frequencies = rfftfreq(signal.size, d=20e-3 / signal.size)
        # remove values outside the threshold
        fourier[frequencies > self.threshold] = 0
        # compute the inverse of rfftn and create a new column
        df[f'{self.x_column}_FFT'] = irfft(fourier)

    def plot_transformed(self, figsize=(7, 7)):
        """
        Plot of the data before and after smoothing.

        Parameters:
            figsize (list), default=(7,7): Figure size.
        """
        # get columns
        columns = self.new_df.columns

        # if the method is lowess plot x and y columns
        if self.method == 'lowess':
            plt.figure(figsize=figsize)
            # plot old values as scatterplot
            scatterplot(self.new_df[self.x_column], self.new_df[self.y_column], label='Raw')
            # plot new values
            if self._lowess_shape == 2:
                plt.plot(self.new_df[columns[-2]], self.new_df[columns[-1]], label=self.method)
            else:
                plt.plot(self.new_df[columns[-1]], self.new_df[self.y_column], label=self.method)
            plt.legend()

        else:
            plt.figure(figsize=(15, 10))
            # plot old x
            plt.plot(self.new_df[self.x_column], label='Raw')
            # plot new x
            plt.plot(self.new_df[columns[-1]], label=self.method)
            plt.legend()
            plt.show()

    def _init_methods_dict(self):
        """
        Methods dictionary initialization.
        """
        self.methods_dict = {
            'moving_average': self._moving_average,
            'lowess': self._lowess,
            's_g_filter': self._savitzky_golaay,
            'fft': self._fft,
        }
