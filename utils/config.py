from pydantic import BaseModel, Field, field_validator
from typing import Literal, Union, Callable, List, Optional
import pandas as pd

class DfConfig(BaseModel):
    value: Literal['text', 'num'] = 'text'
    col_name: Literal['code', 'value']
    col_type: Literal['multi', 'single'] = 'multi'
    melt: bool = False
    loop_on: Optional[str] = None
    loop_mode: Literal['part', 'stack'] = 'part'
    dropna_col: List[str] = Field(default_factory=list)
    
    def to_default(self):
        self.value = 'text'
        self.col_name = 'code'
        self.melt = False
        self.loop_on = None
        
class CtabConfig(BaseModel):
    perc: bool = False
    total: bool = False
    round_perc: bool = True
    cat_aggfunc: Union[str, Callable, List[Union[Callable, str]]] = pd.Series.nunique
    num_aggfunc: List[Union[str, Callable]] = ['mean', 'median', 'count', 'min', 'max', 'std', 'var']
    sig: Union[float, None] = Field(None, ge=0, le=1, allow_none=True)
    dropna: bool=False

    @field_validator('sig')
    def validate_sig(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Value must be between 0 and 1, or None')
        return v
    
    def to_default(self):
        self.perc = False
        self.total = False
        self.round_perc: bool = True
        self.cat_aggfunc = pd.Series.nunique
        self.num_aggfunc = ['mean', 'median', 'count', 'min', 'max', 'std', 'var']
        self.sig = None
        self.dropna = False
    
    @property
    def format(self):
        return {'total': self.total,
                'perc': self.perc,
                'cat_aggfunc': self.cat_aggfunc,
                'num_aggfunc': self.num_aggfunc,
                'sig': self.sig}

    @property
    def cat_format(self) -> dict:
        return {'total': self.total,
                'perc': self.perc,
                'round_perc': self.round_perc,
                'cat_aggfunc': self.cat_aggfunc,
                'sig': self.sig,
                'dropna': self.dropna}

class SpssConfig(BaseModel):
    perc: bool = False
    std: bool = False
    compare_tests: List[str] = ['MEAN', 'PROP']
    alpha: float = 0.1

    def to_default(self):
        self.perc = False
        self.std = False
        self.compare_tests = ['MEAN', 'PROP']
        self.alpha = 0.1

