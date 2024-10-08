from pydantic import BaseModel
from typing import Union, List, Optional, Tuple, Dict
import pandas as pd
from ..question import MultipleAnswer, SingleAnswer, Number, Rank, Response
from ...utils import report_function, CtabConfig, PptConfig
from copy import deepcopy
from itertools import product
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from .ctab_function import _ctab

BaseType = Union[SingleAnswer, MultipleAnswer, Rank]
QuestionType = Union[SingleAnswer, MultipleAnswer, Rank, Number]

class CrossTab(BaseModel):
    bases: List[BaseType]
    targets: List[QuestionType] = []
    config: CtabConfig = CtabConfig()
    ppt_config: PptConfig = PptConfig()
    _dataframe: Optional[pd.DataFrame] = None

    def reset(self):
        self._dataframe = None
        self.config.to_default()
    
    def __and__(self, target=Union[List[QuestionType], QuestionType]):
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
                if not self.config.deep_by:
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
                else:
                    ctab.config.deep_by = self.config.deep_by
                    for k, v in ctab._deep_parts.items():
                        df = v['ctab']
                        title = '_'.join(k.split('[SPLIT]')) + ':' + str(ctab.title)
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

    @property
    def dataframe(self) -> pd.DataFrame:
        if self._dataframe is None:
            self._dataframe = self._get_dataframe()
        return self._dataframe

    def _get_dataframe(self) -> pd.DataFrame:
        if self.config.deep_by:
            parts = self._deep_parts
            dfs = []
            for k, v in parts.items():
                df = v['ctab']
                col_list = v['col_list']
                col_roots = v['col_root']
                names = col_roots + df.columns.names
                cols = pd.MultiIndex.from_tuples([tuple(col_list) +  i for i in df.columns], names=names)
                df.columns = cols
                dfs.append(df)
            return pd.concat(dfs, axis=1)
        else:
            return _ctab(self.config, self.bases, self.targets)
        
    @property
    def _deep_parts(self) -> Dict[str, pd.DataFrame]:
        if not self.config.deep_by:
            raise ValueError('Need to set config: deep_by to take deep_parts')

        # Chuẩn bị các cặp response
        if len(self.config.deep_by) > 1:
            response_pairs = _create_pairs([q.responses for q in self.config.deep_by])
        else:
            response_pairs = [(response,) for response in self.config.deep_by[0].responses]

        # Hàm quản lý đa tiến trình
        def parallel_process(response_pairs, base, target, config):
            with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as process_executor:
                # Sử dụng đa tiến trình để xử lý các cặp
                process_with_params = partial(_process_pair, bases=base, targets=target, config=config)
                results = list(process_executor.map(process_with_params, response_pairs))
            return dict(results)

        # Sử dụng kết hợp đa tiến trình và đa luồng
        with ThreadPoolExecutor() as executor:
            # Kết hợp đa luồng để quản lý các tiến trình nhỏ hơn
            future = executor.submit(parallel_process, response_pairs, self.bases, self.targets, self.config)
            result = future.result()  # Chờ kết quả từ tiến trình chính

        return result
    
def _create_pairs(list_of_lists):
    return list(product(*list_of_lists))

# Hàm filter responses cho từng pair
def _filter_by_responses(questions: List[QuestionType], response_pair: Tuple[Response]):
    # Tránh dùng deepcopy quá nhiều nếu có thể
    questions = deepcopy(questions)  # Tùy vào yêu cầu có thể tối ưu ở đây
    valid_respondents = set()
    for response in response_pair:
        valid_respondents.update(response.respondents)
    
    for question in questions:
        for response in question.responses:
            response.respondents = [r for r in response.respondents if r in valid_respondents]
    return questions

# Hàm xử lý cho từng pair
def _process_pair(pair, bases, targets, config):
    bases = _filter_by_responses(bases, pair)
    targets = _filter_by_responses(targets, pair)
    crosstab = _ctab(config, bases, targets)  # Giả sử self._ctab là hàm tính toán tốn tài nguyên
    key = '[SPLIT]'.join([response.code for response in pair])
    col_list = [response.value for response in pair]
    return key, {
        'ctab': crosstab,
        'col_list': col_list,
        'col_root': [response.root for response in pair]
    }


