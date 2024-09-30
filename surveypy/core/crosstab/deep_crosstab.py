from pydantic import BaseModel
from typing import List, Union, Optional, Callable
import pandas as pd

from ..question import MultipleAnswer, SingleAnswer, Number, Rank
BaseType = Union[SingleAnswer, MultipleAnswer, Rank]
TargetType = Union[SingleAnswer, MultipleAnswer, Rank, Number]


class DeepCrossTab(BaseModel):
    bases: List[List[BaseType]]
    targets: List[List[TargetType]] = []
    _dataframe: Optional[pd.DataFrame] = None
    
    @property
    def dataframe(self) -> pd.DataFrame:
        if self._dataframe is None:
            self._dataframe = self._get_dataframe()
        return self._dataframe

    def _get_dataframe(self) -> pd.DataFrame:
        pass
    
    
    
def _deep_sm_ctab(
        bases: List[BaseType], targets: List[TargetType], 
        total:bool, perc:bool, round_perc=bool,
        cat_aggfunc:Union[Callable, str] = pd.Series.nunique,
        sig=None,
        dropna=False
    ) -> pd.DataFrame:
    
    for question in bases + targets:
        question.df_config.melt = True
        question.df_config.value = 'text'
        
    
