from ..question import MultipleAnswer, SingleAnswer, Number, Rank, Response
from ...utils import report_function, CtabConfig, PptConfig
from pydantic import BaseModel
from typing import Union, List, Optional, Tuple, Dict
import pandas as pd
import multiprocessing as mp

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
        args_list = [(self.bases, target, self.config.total, self.config.perc, self.config.round_perc, self.config.deep_by) for target in self.targets]
        with mp.Pool(mp.cpu_count()) as pool:
            dfs = pool.map(_pivot_target_with_args, args_list)
        # Kết hợp các DataFrame trả về
        return pd.concat(dfs, axis=0)
    
def _pivot_target_with_args(args):
    bases, target, total, perc, round_perc, deep_by = args
    return _pivot_target(bases, target, total, perc, round_perc, deep_by)

def _pivot_target(bases: List[BaseType], target: QuestionType, total=True, perc=True, round_perc=True, deep_by=[]):
    args = {'bases': bases, 'target': target, 'total': total, 'perc': perc, 'round_perc': round_perc, 'deep_by': deep_by}
    if isinstance(target, (SingleAnswer, MultipleAnswer)):
        return _pivot_sm(**args)
    elif isinstance(target, Rank):
        return pd.concat([_pivot_sm(bases, sa, total, perc, round_perc, deep_by) for sa in target.decompose()], axis=0)
    elif isinstance(target, Number):
        return _pivot_number(**args)
    
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


        
def _pivot_sm(bases: List[BaseType], target: QuestionType, total=True, perc=True, round_perc=True, sig=None, deep_by: List[BaseType] = []):
    
    df = _custom_merge(bases, target, deep_by)
    
    deep_indexes = [f'deep_answer_{index}' for index in range(1, len(deep_by) + 1)]

    total_label = 'Total'

    pv = pd.pivot_table(df, columns=deep_indexes + ['root', 'answer'], index=['target_root', 'target_answer'], values='resp_id', 
                        aggfunc=pd.Series.nunique, fill_value=0, margins=True, margins_name=total_label)
    
    total_df = pv.loc[[total_label],:]
    

    pv = pv.loc[~pv.index.get_level_values(0).isin([total_label])]
    
    if perc:
        pv = pv.div(total_df.values, axis=1).fillna(0)
        if round_perc:
            pv = pv.map(lambda x: f'{round(x*100)}%')
    
    pv = pd.concat([pv, total_df])

    index_total_label = f"{target.code}_Total"
    column_total_label = "Total"
    pv.rename(columns={total_label: column_total_label}, inplace=True, index={total_label: index_total_label})
    
    if not total:    
        pv = pv.loc[~pv.index.get_level_values(0).isin([index_total_label]),
                    ~pv.columns.get_level_values(0).isin([column_total_label])]

    desired_columns = []
    if deep_by:
        if total:
            desired_columns.append(('Total', '', ''))

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

    missing_cols = list((set(desired_columns)) - set(pv.columns))
    new_columns = pd.DataFrame(0, index=pv.index, columns=missing_cols)

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

def _pivot_number(bases: List[BaseType], target: QuestionType, deep_by: List[BaseType] = []):
    df = _custom_merge(bases, target)
    
    deep_indexes = [f'deep_answer_{index}' for index in range(1, len(deep_by) + 1)]


    pv = pd.pivot_table(
    df,
    values='target_answer',
    columns=['target_core'],
    index=deep_indexes + ['root', 'answer'],
    aggfunc=['mean', 'median', 'count', 'min', 'max', 'std', 'var'],
    fill_value=0,
    dropna=True
    ).T
    desired_columns = []
    if deep_by:
        for deep in deep_by:
            for deep_response in deep.responses:
                for base in bases:
                    for response in base.responses:
                        desired_columns.append((deep_response.value, base.code, response.value))
    else:
        for base in bases:
            for response in base.responses:
                desired_columns.append((base.code, response.value))

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
