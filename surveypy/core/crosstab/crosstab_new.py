from ..question import MultipleAnswer, SingleAnswer, Number, Rank, Response
from ...utils import report_function, CtabConfig, PptConfig
from pydantic import BaseModel
from typing import Union, List, Optional, Tuple, Dict
import pandas as pd

BaseType = Union[SingleAnswer, MultipleAnswer, Rank]
QuestionType = Union[SingleAnswer, MultipleAnswer, Rank, Number]

class CrossTab(BaseModel):
    bases: List[BaseType]
    targets: List[QuestionType] = []
    config: CtabConfig = CtabConfig()
    ppt_config: PptConfig = PptConfig()
    _dataframe: Optional[pd.DataFrame] = None


    
    
    
    


def _pivot_target(bases: List[BaseType], target: QuestionType, total=True, perc=True, round_perc=True):
    if isinstance(target, (SingleAnswer, MultipleAnswer)):
        return _pivot_sm(bases, target, total, perc, round_perc)
    elif isinstance(target, Rank):
        return pd.concat([_pivot_sm(sa) for sa in target.decompose()], axis=0)
    elif isinstance(target, Number):
        pass
    
def _custom_merge(bases: List[BaseType], target: QuestionType):
    for q in [target] + bases:
        q.df_config.melt = False
        q.df_config.value = 'text'
    df = pd.concat([i.dataframe for i in bases], axis=1)
    df = df.stack(dropna=True).stack(dropna=True).reset_index()
    df.columns = ['resp_id', 'core', 'root', 'answer']
    df = df.query('answer != 0')
    target_df = target.dataframe.stack().reset_index()
    target_df.columns = ['resp_id', 'target_core', 'target_answer']
    target_df = target_df.query('target_answer != 0')

    target_df.loc[:, ['target_root']] = target_df['target_core'].apply(lambda x: '_'.join(x.split('_')[:-1]) if '_' in x else x)

    df = df.merge(target_df, on='resp_id', how='inner')
    
    return df
        
def _pivot_sm(bases: List[BaseType], target: QuestionType, total=True, perc=True, round_perc=True):
    
    df = _custom_merge(bases, target)

    total_label = 'Total'

    pv = pd.pivot_table(df, columns=['root', 'answer'], index=['target_root', 'target_answer'], values='resp_id', 
                        aggfunc=pd.Series.nunique, fill_value=0, margins=True, margins_name=total_label)
    

    total_df = pv.loc[[total_label],:]

    pv = pv.loc[~pv.index.get_level_values(0).isin([total_label])]

    pv = pv.div(total_df.values, axis=1)
    pv = pv.fillna(0)

    pv = pd.concat([pv, total_df])

    index_total_label = f"{target.code}_Total"
    column_total_label = "Total"
    pv.rename(columns={total_label: column_total_label}, inplace=True, index={total_label: index_total_label})

    if not total:    
        pv = pv.loc[~pv.index.get_level_values(0).isin([index_total_label]),
                    ~pv.columns.get_level_values(0).isin([column_total_label])]

    desired_columns = []
    for base in bases:
        for response in base.responses:
            desired_columns.append((base.code, response.value))

    missing_cols = (set(desired_columns)) - set(pv.columns)
    for col in missing_cols:
        pv[col] = 0

    desired_indexes = [response.value for response in target.responses]
    current_indexes = pv.index.get_level_values(1)
    for idx in desired_indexes:
        if idx not in current_indexes:
            # Thêm các hàng mới với giá trị mặc định là 0 cho các index không có
            new_index = pd.MultiIndex.from_tuples([(target.code, idx)], names=pv.index.names)
            new_row = pd.DataFrame([[0] * len(pv.columns)], columns=pv.columns, index=new_index)
            pv = pd.concat([pv, new_row])

    pv = pv.reindex(columns=pd.MultiIndex.from_tuples(desired_columns))
    pv = pv.sort_index(level=1, key=lambda x: pd.Categorical(x, categories=desired_indexes, ordered=True))

    return pv

def _pivot_number(bases: List[BaseType], target: QuestionType):
    df = _custom_merge(bases, target)

    pass
    
