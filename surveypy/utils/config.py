from pydantic import BaseModel, Field, field_validator
from typing import Literal, Union, Callable, List, Optional
import pandas as pd
from pptx.dml.color import MSO_THEME_COLOR
from pptx.enum.chart import XL_LEGEND_POSITION, XL_DATA_LABEL_POSITION


class DfConfig(BaseModel):
    value: Literal['text', 'num'] = 'text'
    col_name: Literal['code', 'value'] = 'code'
    col_type: Literal['multi', 'single'] = 'multi'
    melt: bool = False
    loop_on: List[str] = [None]
    loop_mode: Literal['long', 'wide'] = 'wide'
    
    def to_default(self):
        default_instance = type(self)()
        self.__dict__.update(default_instance.__dict__)
        
class CtabConfig(BaseModel):
    perc: bool = True
    total: bool = False
    round_perc: bool = True
    cat_aggfunc: Union[str, Callable, List[Union[Callable, str]]] = pd.Series.nunique
    num_aggfunc: List[Union[str, Callable]] = ['mean', 'median', 'count', 'min', 'max', 'std', 'var']
    alpha: Union[float, None] = Field(None, ge=0, le=1, allow_none=True)
    adjust: Literal['none', 'bonferroni'] = 'none'
    dropna: bool=False
    deep_by: List = []

    @field_validator('alpha')
    def validate_alpha(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Value must be between 0 and 1, or None')
        return v
    
    @field_validator('deep_by')
    def validate_deep_by(cls, v):
        from ..core.question import SingleAnswer, MultipleAnswer, Rank
        for question in v:
            if not isinstance(question, (SingleAnswer, MultipleAnswer, Rank)):
                raise ValueError('Deep by requires SingleAnswer, MultipleAnswer or Rank')
        return v
    
    def to_default(self):
        default_instance = type(self)()
        self.__dict__.update(default_instance.__dict__)
    
    @property
    def format(self):
        return {'total': self.total,
                'perc': self.perc,
                'cat_aggfunc': self.cat_aggfunc,
                'num_aggfunc': self.num_aggfunc,
                'alpha': self.alpha}

    @property
    def cat_format(self) -> dict:
        return {'total': self.total,
                'perc': self.perc,
                'round_perc': self.round_perc,
                'cat_aggfunc': self.cat_aggfunc,
                'alpha': self.alpha,
                'dropna': self.dropna}
        
class ChartConfig(BaseModel):
    grid: List = [1, 1]
    chart_type: Literal['bar', 'line'] = 'bar'
    barmode: Literal['stack', 'cluster', 'relative', 'overlay'] = 'cluster'
    x_in_base: bool = True
    perc: bool = True
    data_labels: bool = True

class SpssConfig(BaseModel):
    perc: bool = True
    std: bool = False
    compare_tests: List[str] = ['MEAN', 'PROP']
    alpha: float = 0.1
    
    def to_default(self):
        default_instance = type(self)()
        self.__dict__.update(default_instance.__dict__)

class PptConfig(BaseModel):
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
    _legend_position: Literal['top', 'bottom', 'left', 'right', 'top_right'] = 'top'
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
    _data_labels_position: Literal['above', 'below', 'best_fit', 'center', 'inside_base', 'inside_end', 'left', 'mixed', 'outside_end', 'right'] = 'outside_end'
    data_labels_show_category_name: bool = False
    data_labels_show_legend_key: bool = False
    data_labels_show_percentage: bool = False
    data_labels_show_series_name: bool = False
    data_labels_show_value: bool = True
        
    @property
    def legend_position(self):
        mapping = {
            'top': XL_LEGEND_POSITION.TOP,
            'bottom': XL_LEGEND_POSITION.BOTTOM,
            'left': XL_LEGEND_POSITION.LEFT,
            'right': XL_LEGEND_POSITION.RIGHT,
            'corner': XL_LEGEND_POSITION.CORNER,
            'custom': XL_LEGEND_POSITION.CUSTOM
        }
        return mapping[self._legend_position]

    @legend_position.setter
    def legend_position(self, value: Literal['top', 'bottom', 'left', 'right', 'top_right']):
        self._legend_position = value
    
    @property
    def data_labels_position(self):
        mapping = {
            'above': XL_DATA_LABEL_POSITION.ABOVE,
            'below': XL_DATA_LABEL_POSITION.BELOW,
            'best_fit': XL_DATA_LABEL_POSITION.BEST_FIT,
            'center': XL_DATA_LABEL_POSITION.CENTER,
            'inside_base': XL_DATA_LABEL_POSITION.INSIDE_BASE,
            'inside_end': XL_DATA_LABEL_POSITION.INSIDE_END,
            'left': XL_DATA_LABEL_POSITION.LEFT,
            'mixed': XL_DATA_LABEL_POSITION.MIXED,
            'outside_end': XL_DATA_LABEL_POSITION.OUTSIDE_END,
            'right': XL_DATA_LABEL_POSITION.RIGHT
            
        }
        
        return mapping[self._data_labels_position]
    
    @data_labels_position.setter
    def data_labels_position(self, value: Literal['above', 'below', 'best_fit', 'center', 'inside_base', 'inside_end', 'left', 'mixed', 'outside_end', 'right']):
        self._data_labels_position = value
        

    def to_default(self):
        default_instance = type(self)()
        self.__dict__.update(default_instance.__dict__)
