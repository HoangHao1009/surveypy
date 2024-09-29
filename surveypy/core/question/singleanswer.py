from typing import List
from .core_question import Question, Response
import pandas as pd
from ...utils import DfConfig, spss_function
from .utils import _melt_dataframe, _get_duplicates
from concurrent.futures import ThreadPoolExecutor

class SingleAnswer(Question):
    _mode: str = 'normal'

    @property
    def spss_syntax(self):
        code = self.code
        value_label_dict = {index: response.value for index, response in enumerate(self.responses, 1)}
        var_label_command = spss_function.var_label(code, self.text)
        value_label_command = spss_function.value_label(code, value_label_dict)
        return [var_label_command, value_label_command]
    
    @property
    def mode(self):
        return self._mode
    
    @mode.setter
    def mode(self, mode):
        if mode not in ['normal', '1-5', '1-10', '1-4']:
            raise ValueError('MODE must be "normal", "1-4", "1-5" or "1-10"')
        
        if mode != 'normal':
            start_range = min([i.scale for i in self.responses])
            end_range = max([i.scale for i in self.responses])
            range_str = f"{start_range}-{end_range}"
            if mode != range_str:
                raise ValueError(f'Mode setter {mode} is not similar to question range: {range_str}')
            self._mode = mode
        else:
            self._mode = 'normal'
    
    def to_topbottom(self):
        if self.mode == 'normal':
            raise(f'Set question mode to reconstruct')
        elif self.mode == '1-5':
            tb =  self.reconstruct({
                'B2B': [1, 2],
                'Neutral': [3],
                'T2B': [4, 5]
            }, by='scale')
        elif self.mode == '1-10':
            tb = self.reconstruct({
                'B5B': [1, 2, 3, 4, 5],
                'Central': [6, 7, 8],
                'T3B': [8, 9, 10]
            }, by='scale')
        #review spss-syntax creator function
        elif self.mode == '1-4':
            tb = self.reconstruct({
                'B2B': [1, 2],
                'T2B': [3, 4]
            }, by='scale')
        else:
            raise(f'mode {self.mode} can not be reconstruct')
        
        tb.code = f"{self.code}TB"
        return tb
    
    def to_scale(self):
        from .number import Number
        return Number(
            code=f'{self.code}S',
            text=f'{self.text}_Scale',
            type='sa_to_numeric',
            loop_on=self.loop_on,
            responses=self.responses,
            df_config=self.df_config,
        )
    
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
        self.df_config.melt = False
        df = self.dataframe
        df = df.reset_index().groupby(df.columns[-1]).count().reset_index()
        df.columns = ['category', 'count']
        if perc:
            df['count'] = df['count'] / sum(df['count'])
        
        return df
  

def _process_response(code: str, response: Response, config: DfConfig):
    v = response.value if config.value == 'text' else response.scale
    return [{'resp_id': respondent, code: v} for respondent in response.respondents]

