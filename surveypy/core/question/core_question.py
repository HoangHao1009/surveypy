from pydantic import BaseModel, ConfigDict, Field
from typing import List, Dict, Union, Literal, Callable, Optional
import pandas as pd
from copy import deepcopy
from ...utils import DfConfig, report_function, PptConfig
from concurrent.futures import ThreadPoolExecutor, as_completed


class Response(BaseModel):
    code: Optional[str] = None
    value: Union[int, str]
    scale: int
    root: str
    rank: Union[int, None] = 0
    respondents: List[Union[str, int]] = []
    
class Question(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    code: str
    text: str
    type: str
    loop_on: Optional[Union[str, int]] = None
    loop_in_col: bool = True
    responses: List[Response] = []
    df_config: DfConfig = DfConfig()
    ppt_config: PptConfig = PptConfig()
    reconstruct_type: Optional[Literal['classify', 'cluster']] =  None
    reconstruct_mapping: List = []

    @property
    def root(self) -> str:
        split_part = self.code.split('_')
        if 'DET' in split_part or 'OTHER' in split_part:
            return self.code
        elif 'matrix' in self.type:
            return '_'.join(split_part[0:2])
        else:
            return split_part[0]

    @property
    def info(self) -> Dict:
        response_info = [{'code': r.code, 'text': r.value, 'scale': r.scale} for r in self.responses]
        return {
            'root': self.root,
            'code': self.code,
            'type': self.type,
            'loop_on': self.loop_on,
            'num_respondent': len(self.respondents),
            'invalid_respondent': self.invalid,
            'responses': response_info
        }
    
    def __or__(self, other):
        from ..crosstab import CrossTab
        if isinstance(other, list):
            targets = other
        else:
            targets = [other]
        return CrossTab(bases=[self], targets=targets)    

    def __and__(self, other):
        from ..crosstab import CrossTab
        if isinstance(other, list):
            base = [self] + other
        else:
            base = [self, other]
        return CrossTab(bases=base, targets=[])
    
    @property
    def respondents(self):
        result = []
        for response in self.responses:
            result.extend(response.respondents)
        return list(set(result))

    @property
    def _info(self):
        return {'code': self.code, 'text': self.text, 'type': self.type, 'loop_on': self.loop_on}
    
    def get(self, value: str, by: Literal['value', 'scale']='value') -> Response:
        for element in self.responses:
            key = element.value if by == 'value' else element.scale
            if value == key:
                return element
        raise ValueError(f'{value} is not in {self.code}')
    
    def _set_response(self):
        if len(self.responses) == 0:
            raise ValueError(f'Len of {self.code} responses is zero.')
        for index, response in enumerate(self.responses, 1):
            response.code = f'{self.code}_{index}'
            
    def reset(self):
        self._set_response()
        self.df_config.to_default()
        
    def drop(self, value: Union[List, str], by: Literal['value', 'scale']='value', rescale=True, inplace=False):
        if isinstance(value, str):
            value = [value]
        if by == 'value':
            new_responses = [r for r in self.responses if r.value not in value]
        else:
            new_responses = [r for r in self.responses if r.scale not in value]
        if rescale:
            for index, response in enumerate(new_responses, 1):
                response.scale = index
        if inplace:
            self.responses = new_responses
            self._set_response()
        else:
            new_question = deepcopy(self)
            new_question.responses = deepcopy(new_responses)
            new_question._set_response()
            return new_question
        
    def sort(self, response_list: List[str], by: Literal['value', 'scale']='value', rescale=True, inplace=False):
        if by == 'value':
            new_responses = deepcopy(sorted(self.responses, key=lambda obj: response_list.index(obj.value)))
        else:
            new_responses = deepcopy(sorted(self.responses, key=lambda obj: response_list.index(obj.scale)))
        if rescale:
            for index, response in enumerate(new_responses, 1):
                response.scale = index
        if inplace:
            self.responses = new_responses
            self._set_response()
        else:
            new_question = deepcopy(self)
            new_question.responses = new_responses
            new_question._set_response()
            return new_question
        
    def reconstruct(
        self, construct_dict: Dict, 
        method: Literal['cluster', 'classify']='cluster', 
        by: Literal['value', 'scale']='value', 
        new_code: Optional[str] = None, 
        new_type: Optional[str] = None,
        new_text: Optional[str] = None,
        save_dict: bool = False,
    ):
        from .multipleanswer import MultipleAnswer
        from .singleanswer import SingleAnswer
        from .rank import Rank

        if isinstance(self, Rank):
            raise ValueError('Rank can not be reconstructed')
                

        def get_respondents_for_label(old_label):
            try:
                return self.get(old_label, by=by).respondents
            except:
                return []
        
        if method == 'cluster':
            old_labels = [label for labels in construct_dict.values() for label in labels]
            to_ma = len(old_labels) != len(set(old_labels))
            if to_ma:
                print(f'{self.code} - There are overlap labels in construct dict, set to -> "ma"')

            new_responses = []
            with ThreadPoolExecutor() as executor:
                future_to_old_label = {
                    executor.submit(get_respondents_for_label, old_label): old_label 
                    for old_label_list in construct_dict.values() for old_label in old_label_list
                }
                old_label_to_respondents = {}
                for future in as_completed(future_to_old_label):
                    old_label = future_to_old_label[future]
                    old_label_to_respondents[old_label] = future.result()
                    
            mapping = []

            for index, (new_label, old_label_list) in enumerate(construct_dict.items(), 1):
                new_respondents = list(set(r for old_label in old_label_list for r in old_label_to_respondents.get(old_label, [])))
                new_response = Response(code=f'{self.code}_{index}',value=new_label, scale=index, root=self.code, respondents=new_respondents)
                new_responses.append(new_response)
                
                for respondent in new_respondents:
                    for old_label in old_label_list:
                        mapping.append({
                            'resp_id': respondent, 
                            'question_code': self.code, 
                            'answer_code': new_response.code,
                            'answer_text': new_label,
                            'old_answer_text': old_label
                        })
                
            if to_ma or isinstance(self, MultipleAnswer):
                question = MultipleAnswer(**self._info, responses=new_responses)
            else:
                question = SingleAnswer(**self._info, responses=new_responses)
                
        
        elif method == 'classify':
            new_labels = list(set(label for labels in construct_dict.values() for label in labels))
            new_responses = [Response(code=f'{self.code}_{index}', value=new_label, scale=index, root=self.code) for index, new_label in enumerate(new_labels, 1) if not pd.isna(new_label)]

            old_labels = list(construct_dict.keys())
            with ThreadPoolExecutor() as executor:
                future_to_old_label = {executor.submit(self.get, old_label, by=by): old_label for old_label in old_labels}
                old_label_to_respondents = {}
                for future in as_completed(future_to_old_label):
                    old_label = future_to_old_label[future]
                    old_label_to_respondents[old_label] = future.result().respondents
                    
            mapping = []

            for new_response in new_responses:
                for old_label, new_label_list in construct_dict.items():
                    if new_response.value in new_label_list:
                        new_respondents = old_label_to_respondents[old_label]
                        new_response.respondents.extend(new_respondents)
                        
                        for respondent in new_respondents:
                            mapping.append({
                                'resp_id': respondent, 
                                'question_code': self.code, 
                                'answer_code': new_response.code,
                                'answer_text': new_response.value,
                                'old_answer_text': old_label
                            })

            question = MultipleAnswer(**self._info, responses=new_responses)
            

        question.code = new_code if new_code else question.code
        question.type = new_type if new_type else question.type
        question.text = new_text if new_text else question.text
        question.reset()
        
        if save_dict:   
            question.reconstruct_mapping = mapping
        
        question.reconstruct_type = method

        return question
   
    def to_excel(self, excel_path: str, sheet_name: str=None):
        if not sheet_name:
            sheet_name = self.code
        report_function.df_to_excel(self.dataframe, excel_path, sheet_name)

    def to_ppt(self, ppt_path: str, perc: bool=False):
        from .multipleanswer import MultipleAnswer
        from .singleanswer import SingleAnswer
        if isinstance(self, (SingleAnswer, MultipleAnswer)):
            df = self.summarize(perc)
            if df is not None:
                report_function.create_pptx_chart(ppt_path, df, 'pie', title=self.code, config=self.ppt_config)
        else:
            print(f'Question with type {type(self)} can not be to_ppt')

