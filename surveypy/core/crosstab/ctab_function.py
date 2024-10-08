from typing import Union, List, Callable
import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
from ..question import MultipleAnswer, SingleAnswer, Number, Rank
from ...utils import CtabConfig
from copy import deepcopy
from multiprocessing import Pool


BaseType = Union[SingleAnswer, MultipleAnswer, Rank]
QuestionType = Union[SingleAnswer, MultipleAnswer, Rank, Number]

def _ctab(config, bases, targets) -> pd.DataFrame:
    base_dfs = []
    with Pool(processes=4) as pool:  # Sử dụng multiprocessing với 4 tiến trình
        for base in bases:
            if isinstance(base, (SingleAnswer, MultipleAnswer)):
                # Sử dụng starmap để truyền nhiều tham số cho hàm _process_target
                result = pool.starmap(_process_target, [(base, target, config) for target in targets])
            elif isinstance(base, Rank):
                # Sử dụng starmap cho hàm _process_rank
                result = pool.starmap(_process_rank, [(base, target) for target in targets])
            else:
                raise ValueError(f'Invalid base type. Required: SingleAnswer, MultipleAnswer or Rank.')

            # Kết hợp kết quả
            base_dfs.append(pd.concat(result, axis=0))

    return pd.concat(base_dfs, axis=1).fillna(0)

def sig_test(df: pd.DataFrame, sig: float):
    test_df = pd.DataFrame("", index=df.index, columns=df.columns)
    num_tests = int(df.shape[1] * (df.shape[1] - 1) / 2)  # Số lượng phép kiểm định
    if num_tests > 0:
        bonferroni_sig = sig / num_tests  # Mức ý nghĩa sau khi điều chỉnh
    else:
        bonferroni_sig = sig


    # Lặp qua từng hàng của DataFrame
    for index, row in df.iterrows():
        values = row.values  # Lấy các giá trị trong hàng
        # total = np.sum(values)  # Tổng giá trị của hàng đó

        # Lặp qua từng cột và so sánh với các cột khác
        for i in range(len(values)):
            diff_columns = []
            for j in range(len(values)):
                if i != j:
                    group1_count = df.iloc[:, i].sum()
                    group2_count = df.iloc[:, j].sum()
                    total_count = group1_count + group2_count

                    # Kiểm tra nếu tổng của một nhóm bằng 0 (bỏ qua)
                    if group1_count == 0 or group2_count == 0:
                        continue

                    # Thực hiện kiểm định z-test cho tỷ lệ
                    count = np.array([values[i], values[j]])
                    nobs = np.array([total_count, total_count])
                    stat, pval = proportions_ztest(count, nobs)

                    # Nếu p-value nhỏ hơn 0.05 thì ghi nhận sự khác biệt
                    if pval < bonferroni_sig:
                        alphabe_index = j + 65
                        # print(j, df.columns[j], chr(alphabe_index))
                        # diff_columns.append(df.columns[j])
                        diff_columns.append(chr(alphabe_index))

            # Nếu có cột nào khác biệt, thêm ký tự vào ô đó
            if len(diff_columns) > 0:
                test_df.at[index, df.columns[i]] = ''.join(diff_columns)
    return test_df


def _process_target(base, target, config: CtabConfig):
    if isinstance(target, (SingleAnswer, MultipleAnswer)):
        return _sm_ctab(base, target, **config.cat_format)
    elif isinstance(target, Rank):
        return _rank_ctab(base, target, **config.cat_format)
    elif isinstance(target, Number):
        return _num_ctab(base, target, config.num_aggfunc)
    else:
        raise ValueError(f'Target required type: SingleAnswer, MultipleAnswer, Rank or Number. Your input type: {type(target)}: {target}')
    
def _process_rank(base, target, config: CtabConfig):
    if isinstance(target, (SingleAnswer, MultipleAnswer)):
        return _rank_ctab(target, base, **config.cat_format)
    elif isinstance(target, Rank):
        r = []
        for element in base.decompose():
            r.append(_rank_ctab(element, target, **config.cat_format))
        return pd.concat(r, axis=1).fillna(0)
    elif isinstance(target, Number):
        r = []
        for element in base.decompose():
            r.append(_num_ctab(element, target, config.num_aggfunc))
        return pd.concat(r, axis=1, join='inner')
    else:
        raise ValueError(f'Target required type: SingleAnswer, MultipleAnswer, Rank or Number. Your input type: {type(target)}')

def _sm_ctab(
        base:BaseType, target:QuestionType, 
        total:bool, perc:bool, round_perc=bool,
        cat_aggfunc:Union[Callable, str] = pd.Series.nunique,
        sig=None,
        dropna=False
    ) -> pd.DataFrame:
    """
    SingleAnswer-MultipleAnswer crosstab function
    """
    def _custom_merge(base:BaseType, target:QuestionType):
        base = deepcopy(base)
        target = deepcopy(target)
        base.df_config.melt = True
        target.df_config.melt = True
        base.df_config.value = 'text'
        target.df_config.value = 'text'
        
        cross_zero = False
        temp_id = 999999999999
        
        if len(base.respondents) == 0:
            base.responses[0].respondents.append(temp_id)
        if len(target.respondents) == 0:
            target.responses[0].respondents.append(temp_id)
        
        if len(set(base.respondents) & set(target.respondents)) == 0:
            if temp_id not in base.responses[0].respondents:
                base.responses[0].respondents.append(temp_id)
            if temp_id not in target.responses[0].respondents:
                target.responses[0].respondents.append(temp_id)
            cross_zero = True
        
        merge_df = pd.merge(base.dataframe, target.dataframe, on='resp_id')
        
        # if merge_df.shape[0] == 0:
        #     merge_df = pd.concat([merge_df, new_row], ignore_index=True)
            # print('merge shape 0 - base: ', base.responses[0].respondents)
            # print('merge shape 0 - target: ',target.responses[0].respondents)
        
        return merge_df, cross_zero

        
    merge_df, cross_zero = _custom_merge(base, target)

    suffix = '_x' if base.code == target.code else ''

    # if base.type == 'matrix_checkbox':
    #     merge_df[f'{base.code}_core{suffix}'] = merge_df[f'{base.code}_core'].str.rsplit('_', n=1).str[0]
    # if target.type == 'matrix_checkbox':
    #     merge_df[f'{target.code}_core{suffix}'] = merge_df[f'{target.code}_core'].str.rsplit('_', n=1).str[0]

    base_root = f'{base.code}_root{suffix}' if 'matrix' not in base.type else f'{base.code}_core{suffix}'
    base_value = f'{base.code}_value{suffix}'

    target_suffix = '_y' if suffix else ''
    target_root = f'{target.code}_root{target_suffix}' if 'matrix' not in target.type else f'{target.code}_core{suffix}'
    target_value = f'{target.code}_value{target_suffix}'

    index = [target_root, target_value]
    columns = [base_root, base_value]
    index_total_label = f"{target.code}_Total"
    column_total_label = f"{base.code}_Total"

    total_label = 'Total'
    
    pv = pd.pivot_table(
        merge_df,
        values='resp_id',
        index=index,
        columns=columns,
        aggfunc=cat_aggfunc,
        fill_value=0,
        margins=True,
        margins_name=total_label,
        dropna=False
    )
    
    if cross_zero:
        pv.loc[:, :] = 0

    pv.rename_axis(index=['row', 'row_value'], columns=['col', 'col_value'], inplace=True)
    
    total_df = pv.loc[[total_label],:]

    pv = pv.loc[~pv.index.get_level_values(0).isin([total_label])]
    if sig:
        pv_test = pv.loc[:,~pv.columns.get_level_values(0).isin([total_label])]
        test_df = sig_test(pv_test, sig)
    if perc:
        pv = pv.div(total_df.values, axis=1)
        pv = pv.fillna(0)
        if round_perc:
            pv = pv.map(lambda x: f'{round(x*100)}%')
    
    if sig:
        pv = pv.astype(str) + " " + test_df

    pv = pd.concat([pv, total_df])

    pv.rename(columns={total_label: column_total_label}, inplace=True, index={total_label: index_total_label})

    if not total:    
        pv = pv.loc[~pv.index.get_level_values(0).isin([index_total_label]),
                    ~pv.columns.get_level_values(0).isin([column_total_label])]
    
    if not dropna:
        desired_columns = [response.value for response in base.responses]
        desired_indexes = [response.value for response in target.responses]

        missing_cols = (set(desired_columns)) - set(pv.columns.get_level_values(1))

        for col in missing_cols:
            pv[(base.code, col)] = 0

        current_indexes = pv.index.get_level_values(1)
        for idx in desired_indexes:
            if idx not in current_indexes:
                # Thêm các hàng mới với giá trị mặc định là 0 cho các index không có
                new_index = pd.MultiIndex.from_tuples([(target.code, idx)], names=pv.index.names)
                new_row = pd.DataFrame([[0] * len(pv.columns)], columns=pv.columns, index=new_index)
                pv = pd.concat([pv, new_row])

        pv = pv.sort_index(axis=1, level=1, key=lambda x: pd.Categorical(x, categories=desired_columns, ordered=True))
        pv = pv.sort_index(level=1, key=lambda x: pd.Categorical(x, categories=desired_indexes, ordered=True))

    return pv

def _num_ctab(
        base:BaseType, target:Number,
        num_aggfunc: List[Union[Callable, str]]
    ) -> pd.DataFrame:

    base.df_config.melt = True
    target.df_config.melt = True

    merge = pd.merge(base.dataframe, target.dataframe, on='resp_id')

    pv = pd.pivot_table(
        merge,
        values=f'{target.code}_value',
        columns=[f'{target.code}_core'],
        index=[f'{base.code}_root', f'{base.code}_value'],
        aggfunc=num_aggfunc,
        fill_value=0,
        dropna=False
    ).T
    pv.rename_axis(index=['row', 'row_value'], columns=['col', 'col_value'], inplace=True)
    pv.index = pv.index.map(lambda x: (x[-1], x[0]))
    return pv

def _rank_ctab(        
        base:BaseType, target:BaseType, 
        total:bool, perc:bool,
        cat_aggfunc:Union[Callable, str] = pd.Series.nunique,
        sig=None,
        dropna=False
    ) -> pd.DataFrame:
    
    from .crosstab import CrossTab

    if not isinstance(target, Rank) or isinstance(base, Rank):
        raise ValueError('Need base or target is Rank')

    config = CtabConfig(total=total, perc=perc, 
                         cat_aggfunc=cat_aggfunc, sig=sig, dropna=dropna)

    if isinstance(target, Rank):
        ctabs = [CrossTab(bases=[base], targets=[element]) for element in target.decompose()]
        axis = 0
    elif isinstance(base, Rank):
        ctabs = [CrossTab(bases=[element], targets=[base]) for element in base.decompose()] 
        axis = 1

    for ctab in ctabs:
        ctab.config = config
    dfs = [ctab.dataframe for ctab in ctabs] 
    return pd.concat(dfs, axis=axis)