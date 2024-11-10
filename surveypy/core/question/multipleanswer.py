from typing import List
from .core_question import Question, Response
import pandas as pd
from ...utils import DfConfig, spss_function
from .utils import _melt_dataframe, _get_duplicates
from concurrent.futures import ThreadPoolExecutor

class MultipleAnswer(Question):

    @property
    def sub_codes(self):
        return [i.code for i in self.responses]

    @property
    def spss_syntax(self):
        var_label_command = []
        value_label_command = []

        for response in self.responses:
            code = response.code
            label = f'{self.text}_{response.value}'

            var_label_command.append(spss_function.var_label(code, label))
            value_label_command.append(spss_function.value_label(code, {1: response.value}))

        repsonses_code = [i.code for i in self.responses]

        mrset_command = spss_function.mrset(self.code, self.text, repsonses_code)

        return var_label_command + value_label_command + [mrset_command]
    
    @property
    def dataframe(self) -> pd.DataFrame:
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda response: _process_response(self.code, response, self.df_config), self.responses))
            
        flattened_results = [item for sublist in results for item in sublist]

        df = pd.DataFrame(flattened_results)
        
        try:
            df = df.pivot(index='resp_id', columns=self.code, values='value').fillna(0)
        except:
            return None
        df.columns = pd.MultiIndex.from_product([[self.root], df.columns], names=['root', 'core'])
        
        if self.df_config.col_name == 'code':
            desired_columns = [response.code for response in self.responses]
        else:
            desired_columns = [response.value for response in self.responses]
        existing_columns = df.columns.get_level_values(1).tolist()
        missing_columns = [col for col in desired_columns if col not in existing_columns]

        if missing_columns:
            new_columns_df = pd.DataFrame(0, index=df.index, columns=pd.MultiIndex.from_product([[self.root], missing_columns], names=['root', 'core']))
            df = pd.concat([df, new_columns_df], axis=1)
        
        df = df.sort_index(axis=1, level=1, key=lambda x: pd.Categorical(x, categories=desired_columns, ordered=True))
        
        if self.df_config.melt:
            df = _melt_dataframe(self.code, df)
        else:
            if self.loop_on != None and self.loop_in_col:
                df.columns = df.columns.map(lambda x: tuple(f"{i}LOOP{self.loop_on}" for i in x))

        return df.fillna(0)
    
    @property
    def invalid(self):
        invalid = []
        for response in self.responses:
            duplicates = _get_duplicates(response.respondents)
            invalid.extend(duplicates)
        return list(set(invalid))
    
    def summarize(self, perc: bool=False) -> pd.DataFrame:
        self.df_config.col_name = 'code'
        self.df_config.melt = True
        if self.dataframe is not None:
            df = self.dataframe
            df = df.groupby(df.columns[-1]).count().reset_index()
            df = df.loc[:, ['resp_id', f'{self.code}_value']]
            df.columns = ['count', 'category']
            if perc:
                df['count'] = df['count'] / sum(df['count'])
            return df
        else:
            return None

def _process_response(code: str, response: Response, config: DfConfig):
    v = 1 if config.value == 'num' else response.value
    c = response.code if config.col_name == 'code' else response.value
    
    return [{'resp_id': respondent, code: c, 'value': v} for respondent in response.respondents]
