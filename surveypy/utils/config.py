from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Literal, Union, Callable, List, Optional, Any
import pandas as pd
from pptx.dml.color import RGBColor, MSO_THEME_COLOR
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION

class DfConfig(BaseModel):
    value: Literal['text', 'num'] = 'text'
    col_name: Literal['code', 'value'] = 'code'
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

class PptConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    theme_color: List = [
    MSO_THEME_COLOR.ACCENT_1,
    MSO_THEME_COLOR.ACCENT_2,
    MSO_THEME_COLOR.ACCENT_3,
    MSO_THEME_COLOR.ACCENT_4,
    MSO_THEME_COLOR.ACCENT_5,
    MSO_THEME_COLOR.ACCENT_6,
]
    slide_layout: int = 5
    position: List[int] = [2, 2, 7, 4]
    font: str = 'Montserrat'
    has_legend: bool = True
    has_title: bool = True
    legend_position: Any = XL_LEGEND_POSITION.TOP
    legend_font_size: int =  12
    category_axis_has_major_gridlines: bool = False
    category_axis_has_minor_gridlines: bool = False
    category_axis_has_title: bool = False
    category_axis_visible: bool = True
    category_axis_tick_labels_font_size: int = 12
    value_axis_has_major_gridlines: bool = False
    value_axis_has_minor_gridlines: bool = False
    value_axis_visible: bool = False
    data_labels_font_size: int = 8
    data_labels_font: int = 'Montserrat'
    data_labels_number_format: int = 'General'
    data_labels_number_format_is_linked: bool = True
    data_labels_position: Any = XL_LABEL_POSITION.OUTSIDE_END
    data_labels_show_category_name: bool = False
    data_labels_show_legend_key: bool = False
    data_labels_show_percentage: bool = False
    data_labels_show_series_name: bool = False
    data_labels_show_value: bool = True