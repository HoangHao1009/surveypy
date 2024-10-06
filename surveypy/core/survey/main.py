from pydantic import BaseModel, ConfigDict
from typing import Dict, List, Union, Literal
import os
import re
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import pandas as pd
import pyreadstat
from ..question import SingleAnswer, MultipleAnswer, Number, Rank, Response
from ..crosstab import CrossTab
from ...flatform import QuestionPro
from ...utils import report_function, str_function, spss_function, DfConfig, CtabConfig, SpssConfig, PptConfig

QuestionType = Union[SingleAnswer, MultipleAnswer, Number, Rank]

class Survey(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = 'survey'
    questions: List[QuestionType] = []
    df_config: DfConfig = DfConfig()
    ctab_config: CtabConfig = CtabConfig()
    spss_config: SpssConfig = SpssConfig()
    ppt_config: PptConfig = PptConfig()
    control_variables: List = []
    removed: List[str] = []
    resp_info_col: bool = True
    _working_dir: str = ''
    _block_order: List[str] = []

    def reset_config(self):
        self.df_config.to_default()
        self.ctab_config.to_default()
        self.spss_config.to_default()
        self.ppt_config.to_default()

    def reset_question(self):
        for question in self.questions:
            question.df_config = deepcopy(self.df_config)
        self._sort_questions()

    def add(self, add_list: List[QuestionType]):
        self.questions.extend(add_list)
        self._sort_questions()

    def _sort_questions(self):
        unique_qcode = list(set([i.code for i in self.questions]))
        sort_list = sorted(
            unique_qcode, 
            key=lambda code: str_function.custom_sort(code, self.block_order)
        )
        if self.df_config.loop_on is None:
            self.questions = [self[i] for i in sort_list]
        else:
            new_questions = []
            for i in sort_list:
                try:
                    new_questions.append(self[i])
                except:
                    new_questions.extend([self[f'{i}loop{loop}'] for loop in self.loop_list])
            self.questions = new_questions

    @property
    def block_order(self):
        return self._block_order

    @block_order.setter
    def block_order(self, value: List[str]):
        self._block_order = value
        self.reset_question()

    @property
    def working_dir(self):
        working_dir = self._working_dir if self._working_dir else os.getcwd()
        return working_dir
    
    @working_dir.setter
    def working_dir(self, value: str):
        if not os.path.exists(value):
            os.makedirs(value)
        self._working_dir = value

    @property
    def loop_list(self) -> list:
        lst = [question.loop_on for question in self.questions if question.loop_on is not None]
        return list(set(lst))
    
    @property
    def respondents(self) -> list:
        result = []
        for question in self.questions:
            for response in question.responses:
                result.extend(response.respondents)
        return list(set(result))
    
    @property
    def invalid(self) -> list:
        result = []
        for question in self.questions:
            result.extend(question.invalid)
        return list(set(result))
    
    def info(self):
        info = []
        for question in self.questions:
            question_info = {}
            question_info['code'] = question.code
            question_info['loop'] = question.loop_on
            question_info['type'] = question.type
            question_info['invalid'] = question.invalid
            info.append(question_info)
        return info
    
    @property
    def parts(self):
        mapping = {
            'main': ['sa', 'ma', 'sa_matrix', 'ma_matrix', 'rank', 'number'],
            'info': ['respondent_info'],
            'oe': ['text', 'text_dynamic', 'text_other', 'text_matrix']
        }
        
        result = {}
        for i, v in mapping.items():
            questions = [q for q in self.questions if q.type in v]
            survey = self.copy()
            survey.questions = questions
            result[i] = survey
        
        return result
        
    @property
    def question_codes(self):
        return [i.code for i in self.questions]
    
    def copy(self):
        return deepcopy(self)

    def remove(self, type=Literal['question', 'respondent'], remove_list = List[str]):
        if type == 'question':
            self.questions = [i for i in self.questions if i.code not in remove_list]
        elif type == 'respondent':
            for question in self.questions:
                for response in question.responses:
                    new_respondents = [i for i in response.respondents if i not in remove_list]
                    response.respondents = new_respondents
    
    def filter(self, condition_str: str, get_result=False):
        """
        Remove respondents which match condition_str
        """
        operators_map = ['<', '<', '>=', '<=', '==', '!=', 'in']
        for op in operators_map:
            if op in condition_str:
                operation = op
                variable, value = condition_str.split(op)
        variable = variable.strip()  # Loại bỏ khoảng trắng
        filter_question = self[variable]
        filter_question.df_config.melt = True
        query = f"{variable}_value {operation} {value}"
        filter_respondents =  filter_question.dataframe.query(query)['resp_id'].tolist()
        self.remove(type='respondent', remove_list=filter_respondents)
        self.removed.extend(filter_respondents)
        if get_result:
            return filter_respondents
        
    def __getitem__(self, query: Union[str, list]):
        if isinstance(query, str):
            query = query.split('loop')
            code = query[0]
            loop = query[1] if len(query) > 1 else None
            
            for question in self.questions:
                question_code = question.code
                if code == question_code:
                    if question.loop_on == None:
                        return question
                    else:
                        for question in [question for question in self.questions if question.code == code]:
                            if question.loop_on == loop:
                                return question
                        raise KeyError(f'{code} is loop question with loop on: {self.loop_list}. Your input loop is not valid: {loop}')
            question_list = []
            for question in self.questions:
                if code == question.root:
                    question_list.append(question_code)
            additional = ''
            if len(question_list) > 0:
                additional += f'Instead, it is root question with these core question {", ".join(question_list)}'
            raise KeyError(f'{code} is not exists in core questions. ' + additional)
        elif isinstance(query, list):
            if self.df_config.loop_on is None:
                questions = [question for question in self.questions if question.code in query]
            else:
                questions = []
                for var in query:
                    try:
                        question = self[var]
                        questions.append(question)
                    except:
                        question = self[f'{var}loop{self.df_config.loop_on}']
                        questions.append(question)
            
            new = deepcopy(self)
            new.questions = questions
            return new

    def initialize(self, flatform, remove_incomplete: bool=True):
        if not isinstance(flatform, QuestionPro):
            raise ValueError(f'Need QuestionPro/ ... to continue')
        with ThreadPoolExecutor() as executor:
            respondent_questions = list(executor.map(
                lambda item: _process_respondent(*item),
                flatform.respondent_info.items()
            ))
        self.questions.extend(respondent_questions)

        with ThreadPoolExecutor() as executor:
            standard_question = list(executor.map(
                lambda item: _process_question(*item),
                flatform.standard_dict.items()
            ))

        self.questions.extend(
            filter(None, [question for questions in standard_question for question in questions])
        )

        if remove_incomplete:
            self.filter('responseStatus != "Completed"')           

    def recode(
            self, by: Literal['question', 'custom'], 
            question_code: Union[str, None] = None,
            custom_dict: Union[dict, None] = None,
            drop=True, inplace=True
        ):
        def recode_survey(survey):
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(_recode_question, question, recode_dict, drop) for question in survey.questions]
                for future in futures:
                    future.result()

        if question_code and custom_dict:
            print('Required only question_code or custom_dict. You provide both')

        if by == 'question':
            by_question = self[question_code]
            recode_dict = {}
            for response in by_question.responses:
                if len(response.respondents) == 1:
                    recode_dict[response.respondents[0]] = response.value
                elif len(response.respondents) > 1:
                    print(f'{self.name} - Required unique match to recode. Invalid will be drop in {response}')
                else:
                    print(f'{self.name} - Response have no element to match in {response}')
        else:
            recode_dict = custom_dict
        
        if inplace:
            recode_survey(self)
        else:
            from copy import deepcopy
            survey_obj = deepcopy(self)
            recode_survey(survey_obj)
            return survey_obj
        
    def merge(self, target, on: Union[str, dict], by: Literal['question', 'custom']='question'):
        if not isinstance(target, Survey):
            raise ValueError('Required Survey to merge')
        
        existing_codes = {q.code for q in self.questions}
        is_custom = True if by == 'custom' else False
        target_new = target.recode(
            by=by, 
            custom_dict=on if is_custom else None, 
            question_code=on if not is_custom else None, 
            inplace=False
        )

        self.recode(
            by=by, 
            custom_dict=on if is_custom else None, 
            question_code=on if not is_custom else None, 
        )

        for question in target_new.questions:
            if question.code in existing_codes:
                if question.code == on:
                    target_new.remove(type='question', remove_list=[question.code])
                else:
                    self[question.code].code += f'_{self.name}'
                    question.code += f'_{target_new.name}'

        self.questions.extend(target_new.questions)

    @property
    def dataframe(self) -> pd.DataFrame:
        def _process_question(question: QuestionType, loop: str):
            question.df_config.melt = False
            loops = [None] if question.loop_on is None else [None, loop]
            if question.loop_on in loops:
                try:
                    return question.dataframe
                except Exception as e:
                    print(f'Invalid in: Question {question.code} with config: {question.df_config}. Error: {e}')
            return None
        def _process_loop(loop, questions: List[QuestionType]):
            data = []
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(_process_question, question, loop) for question in questions]
                data = [future.result() for future in as_completed(futures) if future.result() is not None]
            part = pd.concat(data, axis=1)
            loop_col = ('Loop', 'Loop')
            part[loop_col] = loop
            reorder_col = [loop_col] + [i for i in part.columns if i != loop_col]
            part = part.loc[:, reorder_col]
            return part
                
        if self.resp_info_col:
            questions = self.questions
        else:
            questions = [question for question in self.questions if question.code not in resp_info]
            
        for question in self.questions:
            question.df_config.col_name = self.df_config.col_name

        if self.df_config.loop_mode == 'part':
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(_process_question, question, self.df_config.loop_on) for question in questions]
                data = [future.result() for future in as_completed(futures) if future.result() is not None]
            
            df = pd.concat(data, axis=1)
        elif self.df_config.loop_mode == 'stack':
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(_process_loop, loop, questions) for loop in self.loop_list]
                parts = [future.result() for future in as_completed(futures)]
            df = pd.concat(parts, axis=0)
            
        value_to_code = {}
        for question in self.questions:
            if isinstance(question, (SingleAnswer, Number)):
                value_to_code[f'{question.root}_{question.code}'] = question.code
            else:
                for response in question.responses:
                    value_to_code[f'{question.code}_{response.value}'] = response.code
        
        def get_sort_key(col):
            if self.df_config.col_name == 'code':
                return col[-1]
            else:
                return value_to_code[f'{col[0]}_{col[-1]}']
        
        sort_columns = sorted(df.columns, key=lambda col: str_function.custom_sort(get_sort_key(col), self.block_order))

        df = df[sort_columns]

        if self.df_config.col_type == 'single':
            df.columns = df.columns.get_level_values(-1)

        if self.df_config.dropna_col:
            df.dropna(subset=self.df_config.dropna_col, inplace = True)
        return df
    
    @property
    def ctab(self) -> CrossTab:
        for i in self.questions:
            i.reset()
        ctab = CrossTab(
            bases = [self[var] for var in self.control_variables],
            targets = [question for question in self.questions]
        )
        ctab.config = self.ctab_config
        return ctab
    
    @property
    def spss_syntaxs(self) -> List[str]:
        syntaxs = []
        for question in self.questions:
            question_copy = deepcopy(question)
            question_copy.code = re.sub(r'[^\w]', 'x', question.code)[0:64]
            for syntax in question_copy.spss_syntax:
                if syntax not in syntaxs:
                    syntaxs.append(syntax)
        if self.control_variables:
            calculate_dict = {}
            for code in self.question_codes:
                try:
                    question = self[code]
                except:
                    question = self[f'{code}loop{self.df_config.loop_on}']
                    
                code = re.sub(r'[^\w]', 'x', code)[0:64]
                    
                if isinstance(question, Number):
                    calculate_dict[code] = ['Mean', 'Std'] if self.spss_config.std else ['Mean']
                else:
                    if isinstance(question, MultipleAnswer):
                        calculate_dict[f'${code}'] = ['ColPct'] if self.spss_config.perc else ['Count']
                    else:
                        calculate_dict[code] = ['ColPct'] if self.spss_config.perc else ['Count']
            by_col = []
            for var in self.control_variables:
                var = f'${var}' if isinstance(self[var], MultipleAnswer) else var
                by_col.append(var)
            ctab_syntax = spss_function.ctab(by_col, calculate_dict, self.spss_config.compare_tests, self.spss_config.alpha)
            syntaxs.append(ctab_syntax)
        syntaxs.append(spss_function.export(f'{self.working_dir}/{self.name}_output.xlsx'))
        return syntaxs
        
    def to_spss(self, folder_path: str=None, dropna: List[str]=[], dataframe=None):
        folder_path = folder_path if folder_path else self.working_dir 
        sav_path = os.path.join(folder_path, f'{self.name}.sav')
        sps_path = os.path.join(folder_path, f'{self.name}.sps')

        if dataframe:
            df = dataframe
        else:
            self.df_config.col_type = 'single'
            self.df_config.value = 'num'
            self.reset_question()
            df = self.dataframe.reset_index().dropna(subset=dropna)
        df.columns = [re.sub(r'[^\w]', 'x', i)[0:64] for i in df.columns] 
        pyreadstat.write_sav(df, sav_path)            
        spss_syntaxs = '\n'.join(self.spss_syntaxs)
        with open(sps_path, 'w') as file:
            file.write(spss_syntaxs)

        self.df_config.to_default()

    def to_excel(self, folder_path: str=None, sheet_name: str=None):
        folder_path = folder_path if folder_path else self.working_dir
        rawdata_excel_path = os.path.join(folder_path, f'{self.name}_raw.xlsx')
        ctab_excel_path = os.path.join(folder_path, f'{self.name}_ctab.xlsx')
        sheet_name = self.name
        report_function.df_to_excel(self.dataframe, rawdata_excel_path, sheet_name)
        report_function.df_to_excel(self.ctab.dataframe, ctab_excel_path, sheet_name)

    def to_ppt(self, folder_path: str=None, template_path: str=None):
        folder_path = folder_path if folder_path else self.working_dir
        ppt_path = os.path.join(folder_path, f'{self.name}.pptx')
        if template_path:
            if not (os.path.exists(template_path) and template_path.endswith('.pptx')):
                raise ValueError('Invalid template path or file format.')
            shutil.copy2(template_path, ppt_path)

        if self.questions:
            for question in self.questions:
                question.ppt_config = self.ppt_config
                question.to_ppt(ppt_path, self.ctab_config.perc)

        if self.control_variables:
            for var in self.control_variables:
                for question in self.questions:
                    ctab = self[var] | question
                    ctab.config = self.ctab_config
                    ctab.ppt_config = self.ppt_config
                    try:
                        ctab.to_ppt(ppt_path)
                    except Exception as e:
                        print(f'{ctab.title} error when to_ppt: {e}')
                        
    def datasets(self, to: Literal['no_return', 'csv', 'excel'] = 'no_return'):
        self.df_config.value = 'text'
        self.reset_question()
        parts = self.parts
        answer_info = defaultdict(list)
        question_info = defaultdict(list)

        for question in self.questions:
            question_info['question_code'].append(question.code)
            question_info['question_type'].append(question.type)
            question_info['question_text'].append(question.text)
            for response in question.responses:
                answer_info['question_code'].append(question.code)
                answer_info['answer_text'].append(response.value)
                answer_info['answer_scale'].append(response.scale)
                answer_info['answer_code'].append(response.code)

        response_data = []
        for question in parts['main'].questions:
            for response in question.responses:
                for respondent in response.respondents:
                    d = {'id': str(respondent) + str(response.code),
                        'resp_id': respondent,
                        'question_code': question.code,
                        # 'answer_text': response.value,
                        # 'answer_scale': response.scale,
                        'answer_code': response.code}
                    response_data.append(d)
                    
        dimResponse = pd.DataFrame(response_data)
        dimAnswer = pd.DataFrame(answer_info)
        dimQuestion = pd.DataFrame(question_info)
        parts['info'].df_config.col_type = 'single'
        dimRespondentInfo = parts['info'].dataframe.reset_index()
        parts['main'].df_config.col_type = 'single'
        dimRespondentChose = parts['main'].dataframe.reset_index()
        
        dimRespondentInfo['timestamp'] = dimRespondentInfo['timestamp'].map(_parse_timestamp)
        dimRespondentInfo['day'] = dimRespondentInfo['timestamp'].dt.day
        dimRespondentInfo['month'] = dimRespondentInfo['timestamp'].dt.month
        dimRespondentInfo['year'] = dimRespondentInfo['timestamp'].dt.year
        dimRespondentInfo['hour'] = dimRespondentInfo['timestamp'].dt.hour
        
        dataset = {
            'dimRespondentInfo': dimRespondentInfo,
            'dimRespondentChose': dimRespondentChose,
            'dimResponse': dimResponse,
            'dimAnswer': dimAnswer,
            'dimQuestion': dimQuestion
        }
        
        if to != 'no_return':
            path = os.path.join(self.working_dir, 'datasets')
            if not os.path.exists(path):
                os.makedirs(path)
            for k, v in dataset.items():
                if to == 'csv':
                    v.to_csv(os.path.join(path, f'{k}.csv'), index=False)
                elif to == 'excel':
                    v.to_excel(os.path.join(path, f'{k}.xlsx'), index=False)
            
        return dataset
    
def _parse_timestamp(timestamp):
    
    # Loại bỏ phần ICT khỏi chuỗi thời gian
    dt_without_tz = " ".join(timestamp.split()[:-1])
    # Chuyển đổi chuỗi thành datetime
    dt = pd.to_datetime(dt_without_tz)
    # Áp dụng lại múi giờ ICT
    return dt.tz_localize('Asia/Bangkok')

         
# def _to_utc(series):
#     # Chuyển đổi thời gian với errors='coerce' để lỗi chuyển thành NaT
#     series = pd.to_datetime(series, format='%d %b, %Y %I:%M:%S %p ICT', errors='coerce')

#     # Chuyển đổi múi giờ từ Asia/Bangkok (ICT) sang UTC
#     series = series.dt.tz_localize('Asia/Bangkok').dt.tz_convert('UTC')

#     # Chuyển đổi sang Unix timestamp
#     series = series.astype(int) // 10**9
#     return series

#support function
def _process_respondent(var: str, response_dict: dict) -> List[SingleAnswer]:
    responses = [
        Response(
            value=str(answer) if not isinstance(answer, int) else answer,
            scale=index,
            root=var,
            respondents=response_list
        )
        for index, (answer, response_list) in enumerate(response_dict.items(), 1)
    ]

    question = SingleAnswer(
            code=var.replace(' ', '_'),
            text=var, type='respondent_info',
            loop_on=None, responses=responses
        )
    
    question.reset()
    return question

def _process_question(loop_on: str, question_dict: Dict[str, dict]):
    def process_single_question(question_code: str, all_info: dict, loop_on: str):
        question_code = question_code.replace(' ', '_')
        question_info = all_info['info']
        loop_on_val = None if loop_on in ['None', ''] else loop_on

        info_dict = dict(
            code=question_code,
            text=question_info['text'],
            # type=question_info['type'],
            loop_on=loop_on_val,
            responses=all_info['responses'],
        )
    
            
        # if question_info['type'] in ['multiplechoice_radio', 'multiplechoice_dropdown', 'matrix_slider', 
        #                              'matrix_radio', 'multiplechoice_smiley',
        #                              'text_multiple_row', 'text_single_row', 'matrix_text']:
        #     question_obj = SingleAnswer(**info_dict)
        #     if question_obj.type not in ['text_multiple_row', 'text_single_row', 'matrix_text']:
        #         construct_dict = {option: [option] for option in question_info['options']} if 'options' in question_info else {}
        #         question_obj = question_obj.reconstruct(construct_dict)
        #     else:
        #         for index, response in enumerate(question_obj.responses, 1):
        #             response.scale = index
        if question_info['type'] in ['multiplechoice_radio', 'multiplechoice_dropdown', 
                                     'multiplechoice_smiley', 'matrix_slider', 'matrix_radio']:
            info_dict['type'] = 'sa' if 'matrix' not in question_info['type'] else 'sa_matrix'
            question_obj = SingleAnswer(**info_dict)
            construct_dict = {option: [option] for option in question_info['options']} if 'options' in question_info else {}
            question_obj = question_obj.reconstruct(construct_dict)
            
        elif question_info['type'] in ['text_multiple_row', 'text_single_row', 'matrix_text']:
            info_dict['type'] = 'text' if 'matrix' not in question_info['type'] else 'text_matrix'
            question_obj = SingleAnswer(**info_dict)
            for index, response in enumerate(question_obj.responses, 1):
                response.scale = index
                
        elif question_info['type'] in ['lookup_table']:
            info_dict['type'] = 'sa'
            question_obj = SingleAnswer(**info_dict)
            
        elif question_info['type'] in ['multiplechoice_checkbox', 'matrix_checkbox']:
            info_dict['type'] = 'ma' if 'matrix' not in question_info['type'] else 'ma_matrix'
            construct_dict = {option: [option] for option in question_info['options']} if 'options' in question_info else {}
            question_obj = MultipleAnswer(**info_dict).reconstruct(construct_dict)
            
        elif question_info['type'] == 'numeric_slider':
            info_dict['type'] = 'number'
            question_obj = Number(**info_dict)
        elif question_info['type'] == 'rank_order_dropdown':
            #? Need to reconstruct?
            info_dict['type'] = 'rank'
            question_obj = Rank(**info_dict)
        elif question_info['type'] in ['text_dynamic', 'text_other']:
            info_dict['type'] = question_info['type']
            question_obj = SingleAnswer(**info_dict)
            for index, response in enumerate(question_obj.responses, 1):
                response.scale = index
        else:
            print(f"Question type {question_info['type']} is not processed yet")
            question_obj = None

        if question_obj:
            question_obj.reset()
        return question_obj

    with ThreadPoolExecutor() as executor:
        questions = list(executor.map(
            lambda item: process_single_question(item[0], item[1], loop_on),
            question_dict.items()
        ))

    return [q for q in questions if q is not None]

def _recode_question(question: QuestionType, recode_dict: dict, drop):
    warning = []
    for response in question.responses:
        new_respondents = []
        for respondent in response.respondents:
            new_code = recode_dict.get(respondent, respondent)
            if new_code == respondent:
                warning.append(new_code)
            new_respondents.append(new_code)
        if drop:
            new_respondents = [i for i in new_respondents if i not in warning]
        response.respondents = new_respondents
    return question

resp_info = ['longitude', 'latitude', 'country', 'region', 'timeTaken', 'responseStatus', 'timestamp']

