"""
Explainability Base Classes
------------------------------

TODO
"""

from ..models.forecasting_model import ForecastingModel
#from ..models.torch_forecasting_model import TorchForecastingModel
from typing import Optional, Tuple, Union, Any, Callable, Dict, List, Sequence
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from ..timeseries import TimeSeries
from ..logging import get_logger, raise_log, raise_if_not, raise_if


logger = get_logger(__name__)


class ForecastingModelExplainer(ABC):

    @abstractmethod
    def __init__(self, model: ForecastingModel, past_steps_explained: int):

        if not model._fit_called:
            raise_log(
                ValueError('The model must be fitted before creating a ForecastingModelExplainer.'),
                logger
                )

        if not model.training_series.is_univariate:
            raise_log(
                ValueError('Explainability only works for univariate timeseries for now. Stay tuned.'),
                logger
                )
        
        # if isinstance(model, TorchForecastingModel):
        #     # TODO We will have to generalize with output_chunk_length functions to be explained.
        #     if model.output_chunk_length > 1:
        #         raise_log(
        #             ValueError('The model must have an output_chunk_length of 1.'),
        #             logger
        #         )

        if model._is_probabilistic():
            # TODO: We can probably add explainability to probabilistic models, by taking the mean output.
            raise_log(
                ValueError('Explainability is only available for non-probabilistic models.'),
                logger
                )

        self.model = model
        self.past_steps_explained = past_steps_explained

    @abstractmethod
    def explain_with_timestamp(self, timestamp: Union[pd.Timestamp, int]) -> object:
        """
        For a given timestamp in the past, give the contributtion of the past_steps_explained previous 
        timesteps of the timeseries
        
        """
        pass

    @abstractmethod
    def explain_with_timeseries(self, series: TimeSeries) -> object:
        """
        For a given timeseries input, give the contribution of the past_steps_explained previous 
        timesteps of the timeseries
        
        """
        pass
    
    def test_stationarity(self) -> bool:
        # TODO : for a time serie, if it not stationary, 
        # shap could be wrong as we compare to the average of a distribution
        return True








