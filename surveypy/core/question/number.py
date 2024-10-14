from typing import List
from .core_question import Question, Response
import pandas as pd
from ...utils import DfConfig, spss_function
from .utils import _melt_dataframe, _get_duplicates
from concurrent.futures import ThreadPoolExecutor

class Number(Question):
    @property
    def spss_syntax(self):
        code = self.code
        to_scale = f'VARIABLE LEVEL {code} (SCALE).'
        return [spss_function.var_label(code, self.text), to_scale]

    @property
    def dataframe(self) -> pd.DataFrame:
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda response: _process_response(self.code, response, self.df_config), self.responses))
            
        flattened_results = [item for sublist in results for item in sublist]

        df = pd.DataFrame(flattened_results)

        df.set_index('resp_id', inplace=True)
        df.columns = pd.MultiIndex.from_product([[self.root], df.columns], names=['root', 'core'])

        if self.df_config.melt:
            df = _melt_dataframe(self.code, df)

        return df.fillna(0)
    
    @property
    def invalid(self):
        respondents = []
        for response in self.responses:
            respondents.extend(response.respondents)
        return _get_duplicates(respondents)
    
    def summarize(self, perc: bool=False) -> pd.DataFrame:
        self.df_config.col_name = 'code'
        self.df_config.melt = True
        df = self.dataframe
        df = df.pivot_table(
            values=f'{self.code}_value',
            columns=f'{self.code}_core',
            aggfunc=['max', 'min', 'mean', pd.Series.nunique],
            fill_value=0
        ).T
        return df
    
def _process_response(code: str, response: Response, config: DfConfig):
    return [{'resp_id': respondent, code: float(response.scale)} for respondent in response.respondents]