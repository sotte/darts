"""
Shap wrapper Class
------------------------------

TODO
"""

from .explainability import ForecastingModelExplainer
from darts.models.forecasting_model import ForecastingModel
from typing import Optional, Tuple, Union, Any, Callable, Dict, List, Sequence
from itertools import product
from abc import ABC, abstractmethod
from inspect import signature
import numpy as np
import pandas as pd
import shap

from ..timeseries import TimeSeries
from ..logging import get_logger, raise_log, raise_if_not, raise_if

logger = get_logger(__name__)

class ShapExplainer(ForecastingModelExplainer):

    def __init__(self, model: ForecastingModel, past_steps_explained: int):
        super().__init__(model, past_steps_explained)

        if past_steps_explained > len(model.training_series)+1:
            raise_log(
                ValueError('The length of the timeseries must be at least past_step_explained+1'),
                logger
                )

        self.X = create_shap_X(slicing(model.training_series, past_steps_explained+1))
        self.explainer = shap.KernelExplainer(self.predict_wrapper_shap, self.X)

    def explain_with_timestamp(self, timestamp: Union[pd.Timestamp, int]) -> List:
        return self.explainer.shap_values(self.X.loc[timestamp])

    def explain_with_timeseries(self, series: TimeSeries) -> object:
        return self.explainer.shap_values(pd.DataFrame(series.values().reshape(1, -1)).loc[0])
    
    def predict_wrapper_shap(self, X: np.ndarray) -> np.ndarray:
        o = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            o[i] = self.model.predict(1, series = TimeSeries.from_values(X[i,:]), num_samples=1).values()[0][0]
        return o

def slicing(ts, slice_length):
    """Creates a list of sliced timeseries of length slice_length
    
    Parameters
    ----------
    ts
        An univariate timeseries
    slice_length
        the length of the sliced series

    """
    list_slices = []
    for idx in range(len(ts.time_index)-slice_length):
        list_slices.append(ts.slice(start_ts=ts.time_index[idx], end_ts=ts.time_index[idx+slice_length]))
    return list_slices

def create_shap_X(slices):
    """ Creates the input shap needs from the univariate time series.

    Parameters
    ----------
    A list of timeseries of length past_steps_explained+1
    """
    X = pd.DataFrame()
    
    for sl in slices:
        X = X.append(pd.DataFrame(sl.values().reshape(1, -1), index=[sl.time_index[-1]]))
    
    return X.drop(X.columns[-1], axis=1)


