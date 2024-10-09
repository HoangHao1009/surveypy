from ..question import MultipleAnswer, SingleAnswer, Number, Rank
from ...utils import report_function, CtabConfig, PptConfig
from pydantic import BaseModel
from typing import Union, List
import pandas as pd
import multiprocessing as mp
import itertools
from .ctab_function import _pivot_target_with_args
import concurrent.futures


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
    
    @property
    def title(self) -> Union[str, list]:
        base_str = ' '.join([base.code for base in self.bases])
        target_str = ' '.join([target.code for target in self.targets])
        return f'{base_str} x {target_str}'

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
        self.config.total = False
        self.config.sig = None
        self.config.round_perc = False
        df = self.dataframe

        def create_chart(row, column, title):
            try:
                part_df = df.loc[row, column].reset_index()
                part_df.rename({'target_answer': 'row_value'}, inplace=True, axis=1)
                report_function.create_pptx_chart(
                    template_path=ppt_path,
                    dataframe=part_df,
                    type='column',
                    title=title,
                    config=self.ppt_config
                )
            except Exception as e:
                print(f'Error creating chart for {row} x {column}: {e}')

        tasks = []

        if self.config.deep_by:
            deep_responses = [deep.responses for deep in self.config.deep_by]
            for pair in itertools.product(*deep_responses):
                for base in self.bases:
                    for target in self.targets:
                        column = pair + (base.code,)
                        title = f'{target.code} x {column}'
                        tasks.append((target.code, column, title))
        else:
            for base in self.bases:
                for target in self.targets:
                    column = base.code
                    title = f'{target.code} x {column}'
                    tasks.append((target.code, column, title))

        # Sử dụng ThreadPoolExecutor để tạo biểu đồ song song
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(create_chart, row, column, title): (row, column) for row, column, title in tasks}
            for future in concurrent.futures.as_completed(futures):
                row, column = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f'Error processing {row} x {column}: {e}')


    # def to_ppt(self, ppt_path: str):
    #     self.config.total = False
    #     self.config.sig = None
    #     self.config.round_perc = False
    #     df = self.dataframe
    #     if self.config.deep_by:
    #         deep_repsonses = [[i.value for i in deep.responses] for deep in self.config.deep_by]
    #         pairs = list(itertools.product(*deep_repsonses))
    #         for pair in pairs:
    #             for base in self.bases:
    #                 for target in self.targets:
    #                     column = pair + (base.code, )
    #                     row = target.code
    #                     title = f'{row} x {column}'
    #                     try:
    #                         part_df = df.loc[row, column].reset_index()
    #                         part_df.rename({'target_answer': 'row_value'}, inplace = True, axis=1)
    #                         report_function.create_pptx_chart(
    #                             template_path=ppt_path,
    #                             dataframe=part_df,
    #                             type='column',
    #                             title=title,
    #                             config=self.ppt_config
    #                         )
    #                     except:
    #                         print(f'{base.code}x{target.code}x{pair} Error to ppt')

    #     else:
    #         for base in self.bases:
    #             for target in self.targets:
    #                 column = base.code
    #                 row = target.code
    #                 title = f'{row} x {column}'
    #                 try:
    #                     part_df = df.loc[row, column].reset_index()
    #                     part_df.rename({'target_answer': 'row_value'}, inplace = True, axis=1)
    #                     report_function.create_pptx_chart(
    #                         template_path=ppt_path,
    #                         dataframe=part_df,
    #                         type='column',
    #                         title=title,
    #                         config=self.ppt_config
    #                     )
    #                 except:
    #                     print(f'{base.code}x{target.code} Error to ppt')

    
