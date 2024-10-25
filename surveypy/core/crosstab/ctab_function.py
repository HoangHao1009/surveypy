from ..question import MultipleAnswer, SingleAnswer, Number, Rank, Response
from ...utils import CtabConfig
from typing import Union, List, Tuple, Dict
import pandas as pd
import itertools
import numpy as np
from copy import deepcopy
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
np.seterr(divide='ignore', invalid='ignore')



BaseType = Union[SingleAnswer, MultipleAnswer, Rank]
QuestionType = Union[SingleAnswer, MultipleAnswer, Rank, Number]

def _custom_merge(bases: List[BaseType], target: QuestionType, deep_by: List[BaseType] = []):
    for q in [target] + bases + deep_by:
        q.df_config.melt = False
        q.df_config.value = 'text'

    df = pd.concat([base.dataframe for base in bases], axis=1).stack(dropna=True).stack(dropna=True).reset_index()
    df.columns = ['resp_id', 'core', 'root', 'answer']
    df = df[df['answer'] != 0]  # Lọc ra các giá trị khác 0 cho cột answer

    target_df = target.dataframe.stack().stack().reset_index()
    target_df.columns = ['resp_id', 'target_core', 'target_root', 'target_answer']
    target_df = target_df[target_df['target_answer'] != 0]
    # target_df = target.dataframe.stack().reset_index()
    # target_df.columns = ['resp_id', 'target_core', 'target_answer']
    # target_df = target_df.assign(target_root=target_df['target_core'].str.rsplit('_', n=1).str[0])

    df = df.merge(target_df, on='resp_id', how='inner')

    for index, deep in enumerate(deep_by, 1):
        deep_df = deep.dataframe.stack().reset_index()
        deep_df.columns = ['resp_id', f'deep_core_{index}', f'deep_answer_{index}']
        df = df.merge(deep_df, on='resp_id', how='inner')

    return df

def _desired_columns(deep_by, total, bases):
    desired_columns = []
    if deep_by:
        if total:
            total_label = ('Total', '')
            for i in range(len(deep_by)):
                total_label += ('', )
            desired_columns.append(total_label)
        for base in bases:
            deep_repsonses = [[i.value for i in deep.responses] for deep in deep_by]
            deep_repsonses.append([base.code])
            deep_repsonses.append([i.value for i in base.responses])
            pairs = list(itertools.product(*deep_repsonses))
            desired_columns.extend(pairs)
    else:
        if total:
            desired_columns.append(('Total', ''))
        for base in bases:
            for response in base.responses:
                desired_columns.append((base.code, response.value))
    return desired_columns

def _df_parts(pv, deep_by, bases) -> Dict:
    result = {}
    deep_repsonses = [[i.value for i in deep.responses] for deep in deep_by]
    pairs = list(itertools.product(*deep_repsonses))
    for pair in pairs:
        for base in bases:
            column = pair + (base.code, )
            try:
                test_df = pv.loc[:, column]
                key = '_'.join(column)
                result[key] = {'column': column, 'df': test_df}
            except:
                continue
    return result


def _pivot_sm(bases: List[BaseType], target: QuestionType, config: CtabConfig):
    df = _custom_merge(bases, target, config.deep_by)

    deep_indexes = [f'deep_answer_{index}' for index in range(1, len(config.deep_by) + 1)]
    total_label = 'Total'

    pv = pd.pivot_table(
        df, 
        columns=deep_indexes + ['root', 'answer'], 
        index=['target_root', 'target_answer'], 
        values='resp_id', 
        aggfunc=config.cat_aggfunc, 
        fill_value=0, 
        margins=True, 
        margins_name=total_label,
        dropna=False
    )
    
    # return pv
    fill = 0 
    
    if not config.dropna:
        desired_columns = _desired_columns(config.deep_by, config.total, bases)
        missing_cols = list(set(desired_columns) - set(pv.columns))
        pv = pd.concat([pv, pd.DataFrame(fill, index=pv.index, columns=missing_cols)], axis=1)
        pv = pv.reindex(columns=pd.MultiIndex.from_tuples(desired_columns))

        desired_indexes = [response.value for response in target.responses]
        missing_indexes = set(desired_indexes) - set(pv.index.get_level_values(1))
        new_rows = pd.DataFrame([[fill] * len(pv.columns)], columns=pv.columns, 
                                index=pd.MultiIndex.from_tuples([(target.code, idx) for idx in missing_indexes], 
                                names=pv.index.names))
                
        pv = pd.concat([pv, new_rows]).sort_index(level=1, key=lambda x: pd.Categorical(x, categories=desired_indexes, ordered=True))
                
    total_df = pv.loc[[total_label],:]
    pv = pv.loc[~pv.index.get_level_values(0).isin([total_label])]
       
    if config.alpha:
        dfs = []
        df_parts = _df_parts(pv, config.deep_by, bases)

        for key, value in df_parts.items():
            column = value['column']
            test_df = value['df']
            test_df.columns = pd.MultiIndex.from_tuples([column + (col,) for col in test_df.columns])
            test_result = _sig_test(test_df, config.alpha, config.perc, config.round_perc)
            dfs.append(test_result)
        final_test = pd.concat(dfs, axis=1)
                
        if config.perc:
            # pv = pv.div(pv.sum(axis=0), axis=1).fillna(0)
            pv = pv.div(total_df.values, axis=1)
            if config.round_perc:
                pv = pv.map(lambda x: f'{round(x*100)}%' if x != 0 and not pd.isna(x) else 0)
                
        if config.total:
            final_test.loc[:, 'Total'] = pv.loc[:, 'Total']

        missing_columns = pv.columns.difference(final_test.columns)
        for col in missing_columns:
            final_test[col] = ''
        final_test = final_test[pv.columns]
        pv = pd.concat([final_test, total_df])
        
    else:
        if config.perc:
            # pv = pv.div(pv.sum(axis=0), axis=1).fillna(0)
            pv = pv.div(total_df.values, axis=1)
            if config.round_perc:
                pv = pv.map(lambda x: f'{round(x*100)}%' if x != 0 and not pd.isna(x) else 0)
        pv = pd.concat([pv, total_df])
        
    pv.rename(columns={total_label: "Total"}, index={total_label: f"{target.code}_Total"}, inplace=True)
    pv = pv.fillna(0)


    if not config.total:
        pv = pv.loc[~pv.index.get_level_values(0).isin([f"{target.code}_Total"]),
                    ~pv.columns.get_level_values(0).isin(["Total"])]
        
    if config.alpha:
        column_letter_mapping = {}
        for q in bases + [target]:
            for response in q.responses:
                if response.value != '':
                    column_letter_mapping[response.value] = str(response.value) + ' ' + f"({chr(64 + int(response.scale))})"
                else:
                    column_letter_mapping[response.value] = ''
    
        pv.rename(columns=lambda x: column_letter_mapping.get(x, ''), level=-1, inplace=True)

    return pv

def _pivot_number(bases: List[BaseType], target: QuestionType, config: CtabConfig):
    df = _custom_merge(bases, target, config.deep_by)
    
    deep_indexes = [f'deep_answer_{index}' for index in range(1, len(config.deep_by) + 1)]

    pv = pd.pivot_table(
    df,
    values='target_answer',
    columns=['target_core'],
    index=deep_indexes + ['root', 'answer'],
    aggfunc=config.num_aggfunc,
    fill_value=0,
    dropna=True
    ).T
    
    pv.index = pd.MultiIndex.from_tuples([(i[-1], i[0]) for i in pv.index], names=['target_root', 'target_answer'])

    fill = 0 

    if not config.dropna:
        desired_columns = _desired_columns(config.deep_by, config.total, bases)
        missing_cols = list(set(desired_columns) - set(pv.columns))
        pv = pd.concat([pv, pd.DataFrame(fill, index=pv.index, columns=missing_cols)], axis=1)
        pv = pv.reindex(columns=pd.MultiIndex.from_tuples(desired_columns))
        
    if config.alpha:
        column_letter_mapping = {}
        for q in bases + [target]:
            for response in q.responses:
                if response.value != '':
                    column_letter_mapping[response.value] = str(response.value) + ' ' + f"({chr(64 + int(response.scale))})"
                else:
                    column_letter_mapping[response.value] = ''
    
        pv.rename(columns=lambda x: column_letter_mapping.get(x, ''), level=-1, inplace=True)
    return pv

def _sig_test(crosstab: pd.DataFrame, alpha: float, perc: bool, round_perc: bool):
    
    num_cols = crosstab.shape[1]

    test_df = pd.DataFrame('', index=crosstab.index, columns=crosstab.columns)

    for i in range(num_cols):
        for j in range(i + 1, num_cols):
            # col1_letter = letters[i]
            # col2_letter = letters[j]
            col1_letter = chr(65 + i)
            col2_letter = chr(65 + j)

            # Lấy số lượng cho từng hàng của cột i và j
            count1 = crosstab.iloc[:, i].values
            count2 = crosstab.iloc[:, j].values

            # Tổng số cho các cột
            n = crosstab.sum(axis=0).values

            p_vals = []
            for row in range(len(count1)):
                current_col1 = count1[row]
                current_col2 = count2[row]
                total_col1 = n[i]
                total_col2 = n[j]
                
                # Kiểm tra nếu tổng bằng 0 để tránh chia cho 0
                if total_col1 == 0 or total_col2 == 0:
                    p_vals.append(np.nan)  # Bỏ qua nếu không có dữ liệu
                    continue                

                col1_proportion = current_col1 / total_col1
                col2_proportion = current_col2 / total_col2
                if current_col1 + current_col2 > 0:  # Kiểm tra có đủ dữ liệu không
                    z_stat, p_val = proportions_ztest([current_col1, current_col2], [total_col1, total_col2])
                    p_vals.append(p_val)
                else:
                    p_vals.append(np.nan)  # Không có dữ liệu cho hàng này

            reject, p_adjusted, _, _ = multipletests(p_vals, method='bonferroni', alpha=alpha)
            for row in range(len(count1)):
                row_reject = reject[row]
                row_p_adjusted = p_adjusted[row]
                if row_reject:
                    if row_p_adjusted < alpha:
                        if col1_proportion > col2_proportion:
                            test_df.iloc[row, i] = f'{col2_letter}'
                        else:
                            test_df.iloc[row, j] = f'{col1_letter}'
                            
    if perc:
        crosstab = crosstab.div(crosstab.sum(axis=0), axis=1)
        if round_perc:
            crosstab = crosstab.map(lambda x: f'{round(x*100)}%' if x != 0 and not pd.isna(x) else 0)
                      
    final_df = crosstab.astype(str) + ' ' + test_df.astype(str)
    return final_df

def _pivot_target(bases: List[BaseType], target: QuestionType, config: CtabConfig):
    if isinstance(target, (SingleAnswer, MultipleAnswer)):
        return _pivot_sm(bases, target, config)
    elif isinstance(target, Rank):
        return pd.concat([_pivot_sm(bases, sa, config) for sa in target.decompose()], axis=0)
    elif isinstance(target, Number):
        return _pivot_number(bases, target, config)
    
def _pivot_target_with_args(args: Tuple[List[BaseType], QuestionType, CtabConfig]):
    bases, target, config = args
    return _pivot_target(bases, target, config)



