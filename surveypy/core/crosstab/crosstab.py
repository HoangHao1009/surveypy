from pydantic import BaseModel
from typing import Union, List, Callable, Optional
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from statsmodels.stats.proportion import proportions_ztest
from ..question import MultipleAnswer, SingleAnswer, Number, Rank
from ...utils import report_function, CtabConfig, PptConfig
from copy import deepcopy
from itertools import product

BaseType = Union[SingleAnswer, MultipleAnswer, Rank]
TargetType = Union[SingleAnswer, MultipleAnswer, Rank, Number]

class CrossTab(BaseModel):
    bases: List[BaseType]
    targets: List[TargetType] = []
    config: CtabConfig = CtabConfig()
    ppt_config: PptConfig = PptConfig()
    _dataframe: Optional[pd.DataFrame] = None
    ####
    ####
    ####
    deep_by: List[BaseType] = [] #use to deep ctab
    
    @property
    def title(self) -> Union[str, list]:
        base_str = ' '.join([base.code for base in self.bases])
        target_str = ' '.join([target.code for target in self.targets])
        return f'{base_str} x {target_str}'
    
    @property
    def dataframe(self) -> pd.DataFrame:
        if self._dataframe is None:
            self._dataframe = self._get_dataframe()
        return self._dataframe
    
    def reset(self):
        self._dataframe = None
        self.config.to_default()
        
    def _get_dataframe(self) -> pd.DataFrame:
        def ctab(bases, targets):
            base_dfs = []
            
            for base in bases:
                if isinstance(base, (SingleAnswer, MultipleAnswer)):
                    with ThreadPoolExecutor() as executor:
                        result = list(executor.map(lambda target: _process_target(base, target, self.config), targets))
                elif isinstance(base, Rank):
                    with ThreadPoolExecutor() as executor:
                        result = list(executor.map(lambda target: _process_rank(base, target), targets))
                else:
                    raise ValueError(f'Invalid base type. Required: SingleAnswer, MultipleAnswer or Rank.')
            
                base_dfs.append(pd.concat(result, axis=0))
            
            return pd.concat(base_dfs, axis = 1).fillna(0)
        
        def create_pairs(list_of_lists):
            if len(list_of_lists) > 1:
                return list(product(*list_of_lists))
            else:
                return list_of_lists
        
        def filter_by_responses(questions, response_pair):
            questions = deepcopy(questions)
            valid_respondents = []
            if isinstance(response_pair, list):
                for response in response_pair:
                    valid_respondents.extend(response.respondents)
            else:
                valid_respondents.extend(response_pair.respondents)
            valid_respondents = list(set(valid_respondents))
            
            for question in questions:
                for response in question.responses:
                    response.respondents = [r for r in response.respondents if r in valid_respondents]
            return questions
        
        if self.deep_by:
        
            response_pairs = create_pairs([q.responses for q in self.deep_by])
            
            dfs = []
            for pair in response_pairs:
                bases = filter_by_responses(self.bases, pair)
                targets = filter_by_responses(self.targets, pair)
                crosstab = ctab(bases, targets)
                cols = [response.code for response in pair]
                    
                final_cols = [cols + list(col) for col in crosstab.columns]
                crosstab.columns = pd.MultiIndex.from_arrays(final_cols)
                dfs.append(crosstab)
            return pd.concat(dfs, axis=1)
    
        else:
            return ctab(self.bases, self.targets)
         
    # NO_DEEP_BY
    # def _get_dataframe(self) -> pd.DataFrame:
    #     base_dfs = []
        
    #     for base in self.bases:
    #         if isinstance(base, (SingleAnswer, MultipleAnswer)):
    #             with ThreadPoolExecutor() as executor:
    #                 result = list(executor.map(lambda target: _process_target(base, target, self.config), self.targets))
    #         elif isinstance(base, Rank):
    #             with ThreadPoolExecutor() as executor:
    #                 result = list(executor.map(lambda target: _process_rank(base, target), self.targets))
    #         else:
    #             raise ValueError(f'Invalid base type. Required: SingleAnswer, MultipleAnswer or Rank.')
        
    #         base_dfs.append(pd.concat(result, axis=0))
        
    #     return pd.concat(base_dfs, axis = 1).fillna(0)
    
    def __and__(self, target=Union[List[TargetType], TargetType]):
        if isinstance(target, list):
            lst = self.targets + target
        elif isinstance(target, (SingleAnswer, MultipleAnswer, Rank, Number)):
            lst = self.targets + [target]
        return CrossTab(
            bases=self.bases,
            targets=lst,
            **self.config.format
        )
    
    def __or__(self, target: Union[List[BaseType], BaseType]):
        if isinstance(target, list):
            lst = self.bases + target
        elif isinstance(target, (SingleAnswer, MultipleAnswer, Rank, Number)):
            lst = self.bases + [target]
        
        return CrossTab(
            bases=lst,
            targets=self.targets,
            **self.config.format
        )

    def to_excel(self, excel_path: str, sheet_name: str=None):
        if not sheet_name:
            sheet_name = 'CrossTab1'
        report_function.df_to_excel(self.dataframe, excel_path, sheet_name)

    def to_ppt(self, ppt_path: str):
        for base in self.bases:
            for target in self.targets:
                ctab = CrossTab(
                    bases=[base],
                    targets=[target],
                    **self.config.format
                )
                df = ctab.dataframe
                title = str(ctab.title)
                df.columns = df.columns.get_level_values(1)
                df.reset_index(level='row_value', inplace=True)

                type = 'bar' if isinstance(target, Number) else 'column'

                report_function.create_pptx_chart(
                    template_path=ppt_path,
                    dataframe=df,
                    type=type,
                    title=title,
                    config=self.ppt_config
                )

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

def _sm_ctab(
        base:BaseType, target:TargetType, 
        total:bool, perc:bool, round_perc=bool,
        cat_aggfunc:Union[Callable, str] = pd.Series.nunique,
        sig=None,
        dropna=False
    ) -> pd.DataFrame:
    """
    SingleAnswer-MultipleAnswer crosstab function
    """
    base.df_config.melt = True
    target.df_config.melt = True
    base.df_config.value = 'text'
    target.df_config.value = 'text'

    merge_df = pd.merge(base.dataframe, target.dataframe, on='resp_id')

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
    
    if pv.shape == [0, 0]:
        print(f'{base} and {target} not have any correspondent')

    pv.rename_axis(index=['row', 'row_value'], columns=['col', 'col_value'], inplace=True)
    total_df = pv.loc[[total_label],:]
    pv = pv.loc[~pv.index.get_level_values(0).isin([total_label])]
    if sig:
        pv_test = pv.loc[:,~pv.columns.get_level_values(0).isin([total_label])]
        test_df = sig_test(pv_test, sig)
    if perc:
        pv = pv.div(total_df.values, axis=1)
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
