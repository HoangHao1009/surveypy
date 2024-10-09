from ..question import MultipleAnswer, SingleAnswer, Number, Rank, Response
from ...utils import report_function, CtabConfig, PptConfig
from pydantic import BaseModel
from typing import Union, List, Optional, Tuple, Dict, Callable
import pandas as pd
import multiprocessing as mp
import itertools
import numpy as np
from statsmodels.stats.proportion import proportions_ztest


BaseType = Union[SingleAnswer, MultipleAnswer, Rank]
QuestionType = Union[SingleAnswer, MultipleAnswer, Rank, Number]

class CrossTab(BaseModel):
    bases: List[BaseType]
    targets: List[QuestionType] = []
    config: CtabConfig = CtabConfig()
    ppt_config: PptConfig = PptConfig()

    @property
    def dataframe(self):
        dfs = []
        args_list = [(self.bases, target, self.config) for target in self.targets]
        with mp.Pool(mp.cpu_count()) as pool:
            dfs = pool.map(_pivot_target_with_args, args_list)
        # Kết hợp các DataFrame trả về
        return pd.concat(dfs, axis=0)
    
def _pivot_target_with_args(args: Tuple[List[BaseType], QuestionType, CtabConfig]):
    bases, target, config = args
    return _pivot_target(bases, target, config)

def _pivot_target(bases: List[BaseType], target: QuestionType, config: CtabConfig):
    
    if isinstance(target, (SingleAnswer, MultipleAnswer)):
        return _pivot_sm(bases, target, config)
    elif isinstance(target, Rank):
        return pd.concat([_pivot_sm(bases, sa, config) for sa in target.decompose()], axis=0)
    elif isinstance(target, Number):
        return _pivot_number(bases, target, config)
    
def _custom_merge(bases: List[BaseType], target: QuestionType, deep_by: List[BaseType] = []):
    for q in [target] + bases + deep_by:
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
        for deep in deep_by:
            for deep_response in deep.responses:
                for base in bases:
                    for response in base.responses:
                        desired_columns.append((deep_response.value, base.code, response.value))
    else:
        if total:
            desired_columns.append(('Total', ''))
        for base in bases:
            for response in base.responses:
                desired_columns.append((base.code, response.value))

    return desired_columns

   
def _pivot_sm(bases: List[BaseType], target: QuestionType, config: CtabConfig):
    
    total = config.total
    perc = config.perc
    deep_by = config.deep_by
    round_perc = config.round_perc
    sig = config.sig
    cat_aggfunc = config.cat_aggfunc
    dropna = config.dropna
    
    
    df = _custom_merge(bases, target, deep_by)
    
    deep_indexes = [f'deep_answer_{index}' for index in range(1, len(deep_by) + 1)]

    total_label = 'Total'

    raw_pv = pd.pivot_table(df, columns=deep_indexes + ['root', 'answer'], index=['target_root', 'target_answer'], values='resp_id', 
                        aggfunc=cat_aggfunc, fill_value=0, margins=True, margins_name=total_label)
    
    total_df = raw_pv.loc[[total_label],:]
    raw_pv = raw_pv.loc[~raw_pv.index.get_level_values(0).isin([total_label])]
    
    if perc:
        fill = '0%'
        pv = raw_pv.div(total_df.values, axis=1)
        pv = pv.fillna(0)
        if round_perc:
            pv = pv.map(lambda x: f'{round(x*100)}%')
    else:
        fill = 0
        pv = raw_pv
        
    if sig:
        deep_repsonses = [[i.value for i in deep.responses] for deep in deep_by]
        pairs = list(itertools.product(*deep_repsonses))
        test_dfs = []
        for pair in pairs:
            for base in bases:
                try:
                    column = pair + (base.code, )
                    test_df = raw_pv.loc[:, column]
                    test_df.columns = pd.MultiIndex.from_tuples([column + (col,) for col in test_df.columns])
                    test_result = _sig_test(test_df, sig)  
                    test_dfs.append(test_result)
                except:
                    pass
        final_test = pd.concat(test_dfs, axis=1)
        missing_columns = pv.columns.difference(final_test.columns)
        for col in missing_columns:
            final_test[col] = ''
        final_test = final_test[pv.columns]

        print('final_test.shape', final_test.shape)
        print('pv.shape', pv.shape)
        print('equal', final_test.columns.equals(pv.columns))
        print('final_test.columns', final_test.columns)
        print('pv.columns', pv.columns)
        print('final_test.index', final_test.index)
        print('pv.index', pv.index)

        pv = pv.astype(str) + " " + final_test  
                     
    pv = pd.concat([pv, total_df])
    index_total_label = f"{target.code}_Total"
    column_total_label = "Total"
    pv.rename(columns={total_label: column_total_label}, inplace=True, index={total_label: index_total_label})
    
    if not total:    
        pv = pv.loc[~pv.index.get_level_values(0).isin([index_total_label]),
                    ~pv.columns.get_level_values(0).isin([column_total_label])]

    if not dropna:
        desired_columns = _desired_columns(deep_by, total, bases)
        missing_cols = list((set(desired_columns)) - set(pv.columns))
        new_columns = pd.DataFrame(fill, index=pv.index, columns=missing_cols)

        # Dùng pd.concat để thêm các cột mới vào DataFrame hiện tại
        pv = pd.concat([pv, new_columns], axis=1)
            
        pv = pv.reindex(columns=pd.MultiIndex.from_tuples(desired_columns))

        desired_indexes = [response.value for response in target.responses]
        current_indexes = pv.index.get_level_values(1)
        for idx in desired_indexes:
            if idx not in current_indexes:
                # Thêm các hàng mới với giá trị mặc định là 0 cho các index không có
                new_index = pd.MultiIndex.from_tuples([(target.code, idx)], names=pv.index.names)
                new_row = pd.DataFrame([[0] * len(pv.columns)], columns=pv.columns, index=new_index)
                pv = pd.concat([pv, new_row])
        pv = pv.sort_index(level=1, key=lambda x: pd.Categorical(x, categories=desired_indexes, ordered=True))
        
        
    return pv

def _pivot_number(bases: List[BaseType], target: QuestionType, config: CtabConfig):
    
    deep_by = config.deep_by
    num_aggfunc = config.num_aggfunc
    dropna = config.dropna

    df = _custom_merge(bases, target)
    
    deep_indexes = [f'deep_answer_{index}' for index in range(1, len(deep_by) + 1)]


    pv = pd.pivot_table(
    df,
    values='target_answer',
    columns=['target_core'],
    index=deep_indexes + ['root', 'answer'],
    aggfunc=num_aggfunc,
    fill_value=0,
    dropna=True
    ).T
    
    if not dropna:
        desired_columns = _desired_columns(deep_by=deep_by, total=False, bases=bases)

        missing_cols = list((set(desired_columns)) - set(pv.columns))
        new_columns = pd.DataFrame(0, index=pv.index, columns=missing_cols)

        # Dùng pd.concat để thêm các cột mới vào DataFrame hiện tại
        pv = pd.concat([pv, new_columns], axis=1)
            
        pv = pv.reindex(columns=pd.MultiIndex.from_tuples(desired_columns))

    return pv

def _sig_test(df: pd.DataFrame, sig: float):
    test_df = pd.DataFrame("", index=df.index, columns=df.columns)
    num_tests = int(df.shape[1] * (df.shape[1] - 1) / 2)  # Số lượng phép kiểm định
    if num_tests > 0:
        bonferroni_sig = sig / num_tests  # Mức ý nghĩa sau khi điều chỉnh
    else:
        bonferroni_sig = sig

    # Lặp qua từng hàng của DataFrame
    for index, row in df.iterrows():
        values = row.values  # Lấy các giá trị trong hàng

        # Lặp qua từng cột và so sánh với các cột khác
        for i in range(len(values)):
            diff_columns = []
            for j in range(len(values)):
                if i != j:
                    # Lấy số lượng của từng nhóm từ hàng hiện tại
                    group1_count = values[i]
                    group2_count = values[j]

                    # Lấy tổng số mẫu của từng nhóm (giả định tổng mẫu của cột là tổng của cả DataFrame)
                    nobs1 = df.iloc[:, i].sum()
                    nobs2 = df.iloc[:, j].sum()
                    
                    # Kiểm tra nếu tổng của một nhóm bằng 0 (bỏ qua)
                    if nobs1 == 0 or nobs2 == 0:
                        continue

                    # Thực hiện kiểm định z-test cho tỷ lệ
                    count = np.array([group1_count, group2_count])
                    nobs = np.array([nobs1, nobs2])
                    stat, pval = proportions_ztest(count, nobs)

                    # Nếu p-value nhỏ hơn mức ý nghĩa đã điều chỉnh, ghi nhận sự khác biệt
                    if pval < bonferroni_sig:
                        alphabe_index = j + 65  # Tạo chữ cái đại diện cho cột khác biệt
                        diff_columns.append(chr(alphabe_index))

            # Nếu có cột nào khác biệt, thêm ký tự vào ô đó
            if len(diff_columns) > 0:
                test_df.at[index, df.columns[i]] = ''.join(diff_columns)
    
    return test_df
