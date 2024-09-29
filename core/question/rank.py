from typing import List
from .core_question import Question, Response
import pandas as pd
from ...utils import DfConfig, spss_function
from .utils import _melt_dataframe, _get_duplicates
from concurrent.futures import ThreadPoolExecutor

class Rank(Question):
    @property
    def spss_syntax(self):
        value_label_dict = {index: response.value for index, response in enumerate(self.responses, 1)}
        var_label_command = []
        value_label_command = []

        for response in self.responses:
            code = response.code
            var_label = f"{self.text}_RANK{response.rank}"
            value_label = spss_function.value_label(code, value_label_dict)
            var_label_command.append(spss_function.var_label(code, var_label))
            value_label_command.append(value_label)
        return var_label_command + value_label_command
    
    def decompose(self):
        from .singleanswer import SingleAnswer
        from .core_question import Response
        rank_set = {i.rank for i in self.responses}
        value_list = list({i.value for i in self.responses})

        elements = [
            SingleAnswer(
                code=f"{self.code}rank{r}",
                text=f"{self.text}_rank{r}",
                type="rank_decomposed",
                loop_on=self.loop_on,
            )
            for r in rank_set
        ]

        for element in elements:
            rank = int(element.code.split('rank')[-1])
            responses_by_value = {resp.value: resp for resp in self.responses if resp.rank == rank}
            for value, resp in responses_by_value.items():
                try:
                    response = element.get(value)
                except:
                    response = None
                if response:
                    response.respondents.append(resp.respondents)
                else:
                    element.responses.append(
                        Response(
                            value=value,
                            scale=value_list.index(value),
                            root=element.code,
                            rank=rank,
                            respondents=resp.respondents
                        )
                    )    
        return elements
    
    @property
    def dataframe(self):
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda response: _process_response(self.code, response, self.df_config), self.responses))
            
        flattened_results = [item for sublist in results for item in sublist]

        df = pd.DataFrame(flattened_results)

        df = df = df.pivot(index='resp_id', columns=self.code, values='value').fillna(0)
        df.columns = pd.MultiIndex.from_product([[self.root], df.columns], names=['root', 'core'])

        if self.df_config.melt:
            df = _melt_dataframe(df)
            
        return df.fillna(0)
    
    @property
    def invalid(self):
        invalid = []
        for sa_question in self.decompose():
            respondents = []
            for response in sa_question.responses:
                respondents.extend(response.respondents)
            invalid.extend(_get_duplicates(respondents))
        return list(set(invalid))
    
    def summarize(self, perc: bool=False) -> pd.DataFrame:
        self.df_config.col_name = 'code'
        self.df_config.melt = True
        df = self.dataframe
        return df.pivot_table(
            values='resp_id',
            index=[f'{self.code}_value'],
            columns=[f'{self.code}_core'],
            aggfunc=pd.Series.nunique,
            fill_value=0
        )

    
def _process_response(code: str, response: Response, config: DfConfig):
    return [{'resp_id': respondent, code: f"{code}RANK{response.rank}", 'value': response.value} for respondent in response.respondents]