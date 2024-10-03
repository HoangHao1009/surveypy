from pydantic import BaseModel, Field, field_validator
from typing import Literal, Union, Callable, List, Optional
import pandas as pd
from pptx.dml.color import MSO_THEME_COLOR
from pptx.enum.chart import XL_LEGEND_POSITION, XL_LABEL_POSITION

class DfConfig(BaseModel):
    value: Literal['text', 'num'] = 'text'
    col_name: Literal['code', 'value'] = 'code'
    col_type: Literal['multi', 'single'] = 'multi'
    melt: bool = False
    loop_on: Optional[str] = None
    loop_mode: Literal['part', 'stack'] = 'part'
    dropna_col: List[str] = Field(default_factory=list)
    
    def to_default(self):
        default_instance = type(self)()
        self.__dict__.update(default_instance.__dict__)
        
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
        default_instance = type(self)()
        self.__dict__.update(default_instance.__dict__)
    
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
    legend_position: Literal['top', 'bottom', 'left', 'right', 'top_right'] = 'top'
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
    data_labels_position: Literal['center', 'inside_end', 'inside_base', 'outside_end', 'best_fit'] = 'outside_end'
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
            'top_right': XL_LEGEND_POSITION.TOP_RIGHT
        }
        
        return mapping[self.legend_position]
    
    @property
    def data_labels_position(self):
        mapping = {
            'center': XL_LABEL_POSITION.CENTER,
            'inside_end': XL_LABEL_POSITION.INSIDE_END,
            'inside_base': XL_LABEL_POSITION.INSIDE_BASE,
            'outside_end': XL_LABEL_POSITION.OUTSIDE_END,
            'best_fit': XL_LABEL_POSITION.BEST_FIT
        }
        
        return mapping[self.data_labels_position]

    def to_default(self):
        default_instance = type(self)()
        self.__dict__.update(default_instance.__dict__)
