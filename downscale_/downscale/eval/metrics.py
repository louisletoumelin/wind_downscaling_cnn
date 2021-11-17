import numpy as np
import pandas as pd


class Metrics:

    def __init__(self):
        pass

    @staticmethod
    def pearson_correlation(y_true, y_pred, **kwargs):
        # return(tf.linalg.trace(tfp.stats.correlation(y_pred, y_true))/3)
        pd_series = pd.core.series.Series

        # Change type if required
        y_true = pd.DataFrame(y_true) if not isinstance(y_true, pd_series) else y_true
        y_pred = pd.DataFrame(y_pred) if not isinstance(y_pred, pd_series) else y_pred

        return pd.concat([y_true, y_pred], axis=1).corr().iloc[0, 1]

    @staticmethod
    def absolute_error(pred, true, **kwargs):

        input_is_dataframe = type(pred) == type(true) == pd.core.frame.DataFrame
        input_is_series = type(pred) == type(true) == pd.core.series.Series

        if input_is_dataframe or input_is_series:
            diff = (pred - true).apply(np.abs)
            return diff
        else:
            pred = np.array(pred)
            true = np.array(true)
            return np.abs(pred-true)

    @staticmethod
    def absolute_error_relative(pred, true, **kwargs):

        input_is_dataframe = type(pred) == type(true) == pd.core.frame.DataFrame
        input_is_series = type(pred) == type(true) == pd.core.series.Series

        if input_is_dataframe or input_is_series:
            diff = (pred - true).apply(np.abs) / true
            return diff
        else:
            pred = np.array(pred)
            true = np.array(true)
            return np.where(true != 0, np.abs(pred-true)/true, np.nan)

    @staticmethod
    def RMSE(pred, true, **kwargs):

        input_is_dataframe = type(pred) == type(true) == pd.core.frame.DataFrame
        input_is_series = type(pred) == type(true) == pd.core.series.Series

        if input_is_dataframe or input_is_series:
            diff = (pred - true) ** 2
            return diff.mean() ** 0.5
        else:
            pred = np.array(pred)
            true = np.array(true)
            return np.sqrt(np.nanmean((pred - true) ** 2))

    @staticmethod
    def mean_bias(pred, true, **kwargs):

        input_is_dataframe = type(pred) == type(true) == pd.core.frame.DataFrame
        input_is_series = type(pred) == type(true) == pd.core.series.Series

        if input_is_dataframe or input_is_series:
            diff = (pred - true)
            return diff.mean()
        else:
            pred = np.array(pred)
            true = np.array(true)
        return np.nanmean(pred - true)

    @staticmethod
    def bias(pred, true, **kwargs):

        input_is_dataframe = type(pred) == type(true) == pd.core.frame.DataFrame
        input_is_series = type(pred) == type(true) == pd.core.series.Series

        if input_is_dataframe or input_is_series:
            diff = (pred - true)
            return diff
        else:
            pred = np.array(pred)
            true = np.array(true)
        return pred - true

    @staticmethod
    def bias_rel(pred, true, **kwargs):

        input_is_dataframe = type(pred) == type(true) == pd.core.frame.DataFrame
        input_is_series = type(pred) == type(true) == pd.core.series.Series

        if input_is_dataframe or input_is_series:
            diff = (pred - true)/true
            return diff
        else:
            pred = np.array(pred)
            true = np.array(true)
        return np.where(true != 0, (pred - true)/true, np.nan)

    def bias_rel_wind_1(self, pred, true, variable=None, min_speed=1):
        """
        Relative bias observations > 1
        """
        input_is_dataframe = type(pred) == type(true) == pd.core.frame.DataFrame
        input_is_series = type(pred) == type(true) == pd.core.series.Series

        if input_is_dataframe or input_is_series:
            pred = pred[true[variable] >= min_speed]
            true = true[true[variable] >= min_speed]
            return self.bias_rel(pred, true)
        else:
            pred = np.array(pred)
            true = np.array(true)
            pred = pred[true >= min_speed]
            true = true[true >= min_speed]
        return self.bias_rel(pred, true)

    def abs_error_rel_wind_1(self, pred, true, variable=None, min_speed=1):
        """
        Relative bias observations > 1
        """
        input_is_dataframe = type(pred) == type(true) == pd.core.frame.DataFrame
        input_is_series = type(pred) == type(true) == pd.core.series.Series

        if input_is_dataframe or input_is_series:
            pred = pred[true[variable] >= min_speed]
            true = true[true[variable] >= min_speed]
            return np.abs(self.bias_rel(pred, true))
        else:
            pred = np.array(pred)
            true = np.array(true)
            pred = pred[true >= min_speed]
            true = true[true >= min_speed]
        return np.abs(self.bias_rel(pred, true))

    def bias_direction(self, pred, true, **kwargs):

        input_is_dataframe = type(pred) == type(true) == pd.core.frame.DataFrame
        input_is_series = type(pred) == type(true) == pd.core.series.Series

        diff1 = np.mod((pred - true), 360)
        diff2 = np.mod((true - pred), 360)

        if input_is_dataframe or input_is_series:
            res = pd.concat([diff1, diff2]).min(level=0)
            return res
        else:
            res = np.min([diff1, diff2], axis=0)
            return res

    def _select_metric(self, metric):
        if metric == "abs_error":
            metric_func = self.absolute_error
        elif metric == "bias":
            metric_func = self.bias
        elif metric == "abs_error_rel":
            metric_func = self.absolute_error_relative
        elif metric == "bias_rel":
            metric_func = self.bias_rel
        elif metric == "bias_rel_wind_1":
            metric_func = self.bias_rel_wind_1
        elif metric == "bias_direction":
            metric_func = self.bias_direction
        else:
            raise NotImplementedError(f"{metric} is not defined")

        return metric_func
