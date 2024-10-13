import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel
from typing import List, Dict
from collections import defaultdict
from ...core.question import Response
from ...utils import str_function

def _fetch(url: str, headers: dict , payload: dict) -> list:
    response = requests.request("GET", url, headers=headers, data=payload)
    return response.json()['response']

def _process_respondent(response: dict, result_dict: dict, respondent_info: list) -> dict:
    id = response['responseID']
    for key, value in response.items():
        if key in respondent_info:
            if key in ['location', 'customVariables']:
                for sub_key, sub_value in value.items():
                    result_dict[sub_key][sub_value].append(id)
            else:
                result_dict[key][value].append(id)
    return result_dict

def _process_question(question: dict):
    question_text = str_function.parse_html(question['text'])
    question_code = question['code']
    question_type = question['type']

    if 'answers' in question:
        try:
            options = [str_function.parse_html(answer['text']) for answer in question['answers']]
        except Exception as e:
            options = None
            print(f'{question_code} - {question_type} [ANSWER] is not have any option. Question text: {question_text}. Error: {e}')
        return {question_code: {'text': question_text, 'type': question_type, 'options': options}}

    elif 'rows' in question:
        result = {}

        if len(question['rows']) == 1:
            try:
                options = [str_function.parse_html(col['text']) for col in question['columns']]
            except Exception as e:
                options = None
                print(f'{question_code} - {question_type} [ROW] is not have any option. Question text: {question_text}')
            result[question_code] = {'text': str_function.parse_html(question['rows'][0]['text']),
                                    'type': question_type, 
                                    'options': options}
            
        elif len(question['rows']) > 1:
            for i, row in enumerate(question['rows'], 1):
                row['text'] = str_function.parse_html(row['text'])
                q_code = f"{question['code']}_{i}"
                q_text = f"{question['text']}_{row['text']}" if question['text'] != '' else row['text']
                try:
                    options = [str_function.parse_html(col['text']) for col in question['columns']]
                except Exception as e:
                    options = None
                    print(f'{q_code} - {question_type} [ROW] is not have any option. Question text: {q_text}')
                result[q_code] = {'text': q_text, 'type': question_type, 'options': options}

        else:
            print(f"Error in {question['code']} with len: {len(question['rows'])}")            
        return result
    
    else:
        print(question['code'], "have not both answers or rows")
        return {question_code: {'text': question_text, 'type': question_type}}
    
def _process_responses(responses: List[Dict], result_dict: defaultdict, question_dict: dict):
    def get_response_obj(key, value, scale, root, rank):
        scale = int(scale) if scale != '' else 0
        for question_key, list_response in list(result_dict.items()):
            if key == question_key:
                for response in list_response:
                    if response.value == value and response.scale == scale and response.root == root and response.rank == rank:
                        return response
        response = Response(
            value=value,
            scale=scale,
            root=root,
            rank=rank
        )
        result_dict[key].append(response)
        return response
    
    for resp in responses:
        for q in resp['responseSet']:
            question_code = q['questionCode']
            for a in q['answerValues']:
                if a['answerText'] != '' and a['value']['text'] != '':
                    question_code = f"{question_code}text{a['answerText']}"
                    value = a['value']['text']
                else:
                    value = a['answerText'] if a['answerText'] != '' else a['value']['text']
                value = str_function.parse_html(value)
                scale = a['value']['scale']
                rank = int(a['value']['rank']) if 'rank' in a['value'] else 0
                root = question_code
                loop_on = a.get('sourceAnswerText', None)
                if loop_on == '':
                    loop_on = a['sourceAnswerID']
                loop_on = str_function.parse_html(loop_on)
                key = f"{question_code}loop{loop_on}"
                response_obj = get_response_obj(key, value, scale, root, rank)
                response_obj.respondents.append(resp['responseID'])

                dynamic_text = a['value'].get('dynamicExplodeText', '')

                if dynamic_text != '' and question_code in question_dict:
                    text = a['answerText'].replace(' ', '')
                    det_code = f"{question_code}_{text}DET"
                    question_dict[det_code] = {}
                    question_dict[det_code]['code'] = det_code
                    question_dict[det_code]['type'] = 'text_dynamic'
                    question_dict[det_code]['text'] = f"{question_code}_{text}_DynamicText"
                    key = f"{det_code}loop{loop_on}"
                    value = str_function.parse_html(dynamic_text)
                    det_response_obj = get_response_obj(key, value, scale, root, rank)
                    det_response_obj.respondents.append(resp['responseID'])

                other_text = a['value'].get('other', '')

                if other_text != '' and question_code in question_dict:
                    text = a['answerText'].replace(' ', '')
                    other_code = f"{question_code}_{text}OTHER"
                    question_dict[other_code] = {}
                    question_dict[other_code]['code'] = other_code
                    question_dict[other_code]['type'] = 'text_other'
                    question_dict[other_code]['text'] = f"{question_code}_{text}_OtherText"
                    key = f"{other_code}loop{loop_on}"
                    value = str_function.parse_html(other_text)
                    other_response_obj = get_response_obj(key, value, scale, root, rank)
                    other_response_obj.respondents.append(resp['responseID'])
                    

class QuestionPro(BaseModel):
    survey_id: str
    api_key: str
    env: str = 'com'
    response_pages: int = 1
    question_pages: int = 1
    question_per_page: int = 100
    question_languageID: int = 250
    response_languageID: int = 250
    respondent_variable: list = ['timestamp', 'location', 'timeTaken', 'responseStatus', 'customVariables']
    _cached_question_info = None
    _cached_response_info = None
    _cached_respondent_info = None
    _cached_standard_dict = None


    @property
    def survey_url(self):
        return f'https://api.questionpro.{self.env}/a/api/v2/surveys/{self.survey_id}'
    
    @property
    def question_info(self):
        if self._cached_question_info is None:
            self._cached_question_info = self._get_question_info()
        return self._cached_question_info

    @property
    def response_info(self):
        if self._cached_response_info is None:
            self._cached_response_info = self._get_response_info()
        return self._cached_response_info

    @property
    def respondent_info(self):
        if self._cached_respondent_info is None:
            self._cached_respondent_info = self._get_respondent_info()
        return self._cached_respondent_info

    @property
    def standard_dict(self):
        if self._cached_standard_dict is None:
            self._cached_standard_dict = self._get_standard_dict()
        return self._cached_standard_dict

    def _get_question_info(self):
        payload = {}
        headers = {
            'api-key': self.api_key
            }

        question_url = self.survey_url + f'/questions?page={self.question_pages}&perPage={self.question_per_page}&languageID={self.question_languageID}'
        return _fetch(question_url, headers, payload)
    
    def _get_response_info(self):
        response_urls = [self.survey_url + f'/responses?page={i+1}&perPage=100&languageID={self.response_languageID}' for i in range(self.response_pages)]
        payload = {}
        headers = {
            'api-key': self.api_key
        }
        with ThreadPoolExecutor() as executor:
            response_json = list(executor.map(lambda url: _fetch(url, headers, payload), response_urls))
        response_json = [item for sublist in response_json for item in sublist]
        return response_json
    
    def _get_respondent_info(self):
        respondent_dict = defaultdict(lambda: defaultdict(list))

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(_process_respondent, response, respondent_dict, self.respondent_variable) for response in self.response_info]
            for future in futures:
                future.result()
        return respondent_dict
    
    def _get_standard_dict(self):
        dict2 = defaultdict(list)
        dict3 = defaultdict(lambda: defaultdict(dict))

        with ThreadPoolExecutor() as excecutor:
            futures = [excecutor.submit(_process_question, question) for question in self.question_info]
            dict1 = {}

            for future in as_completed(futures):
                result = future.result()
                dict1.update(result)

        with ThreadPoolExecutor() as excecutor:
            response_chunks = [self.response_info[i:i + 20] for i in range(0, len(self.response_info), 20)]
            futures = [excecutor.submit(_process_responses, chunk, dict2, dict1) for chunk in response_chunks]
            for future in as_completed(futures):
                future.result()

        for question_key, list_response in dict2.items():
            question_code = question_key.split('loop')[0]
            loop_on = question_key.split('loop')[-1]
            question_info = dict1.get(question_code.split('text')[0], None)
            if question_info:
                dict3[loop_on][question_code]['info'] = question_info
                dict3[loop_on][question_code]['responses'] = list_response
        return dict3

































# class QuestionProOld(BaseModel):
#     survey_id: str
#     api_key: str
#     env: str = 'com'
#     response_pages: int = 1
#     question_pages: int = 1
#     question_per_page: int = 100
#     question_languageID: int = 250
#     response_languageID: int = 250
#     standard_dict: dict = {}

#     @property
#     def survey_url(self):
#         return f'https://api.questionpro.{self.env}/a/api/v2/surveys/{self.survey_id}'
    
#     @property
#     def question_json(self):
#         payload = {}
#         headers = {
#             'api-key': self.api_key
#             }

#         question_url = self.survey_url + f'/questions?page={self.question_pages}&perPage={self.question_per_page}&languageID={self.question_languageID}'
#         return requests.request("GET", question_url , headers=headers, data=payload).json()['response']

#     @property
#     def response_json(self):
#         response_urls = [self.survey_url + f'/responses?page={i+1}&perPage=100&languageID={self.response_languageID}' for i in range(self.response_pages)]
#         payload = {}
#         headers = {
#             'api-key': self.api_key
#             }
#         response_json = []
#         for link in response_urls:
#             response = requests.request("GET", link, headers=headers, data=payload)
#             response_json.extend(response.json()['response'])
#         return response_json
    
#     @property
#     def respondent_dict(self):
#         take = ['timestamp', 'location', 'timeTaken', 'responseStatus', 'externalReference', 'customVariables']
#         dict1 = defaultdict(lambda: defaultdict(list))

#         for response in self.response_json:
#             id = response['responseID']
#             for key, value in response.items():
#                 if key in take:
#                     if key in ['location', 'customVariables']:
#                         for sub_key, sub_value in value.items():
#                             dict1[sub_key][sub_value].append(id)
#                     else:
#                         dict1[key][value].append(id)
#         return dict1

#     def get(self):
#         dict1 = {}
#         for question in self.question_json:
#             question['text'] = str_function.parse_html(question['text'])
#             if 'answers' in question:
#                 question_code = question['code']
#                 question_text = str_function.parse_html(question['text'])
#                 question_type = question['type']
#                 options = []
#                 try:
#                     options = [str_function.parse_html(answer['text']) for answer in question['answers']]
#                 except:
#                     options = None
#                     print(f'{question_code} - {question_type} [ANSWER] is not have any option. Question text: {question_text}')
#                 dict1[question_code] = {
#                     'text': question_text,
#                     'type': question_type,
#                     'options': options
#                 }
#             elif 'rows' in question:
#                 if len(question['rows']) == 1:
#                     question_code = question['code']
#                     question_text = str_function.parse_html(question['rows'][0]['text'])
#                     question_type = question['type']
#                     try:
#                         options = [str_function.parse_html(col['text']) for col in question['columns']]
#                     except:
#                         options = None
#                         print(f'{question_code} - {question_type} [ROW] is not have any option. Question text: {question_text}')

#                     dict1[question_code] = {
#                         'text': question_text,
#                         'type': question_type,
#                         'options': options
#                     }

#                 elif len(question['rows']) > 1:
#                     for i, row in enumerate(question['rows'], 1):
#                         row['text'] = str_function.parse_html(row['text'])
#                         question_code = f"{question['code']}_{i}"
#                         question_text = f"{question['text']}_{row['text']}" if question['text'] != '' else row['text']
#                         question_type = question['type']
#                         try:
#                             options = [str_function.parse_html(col['text']) for col in question['columns']]
#                         except:
#                             options = None
#                             print(f'{question_code} - {question_type} [ROW] is not have any option. Question text: {question_text}')
#                         dict1[question_code] = {
#                             'text': question_text,
#                             'type': question_type,
#                             'options': options
#                         }
#                 else:
#                     print(f"Error in {question['code']} with len: {len(question['rows'])}")
#             else:
#                 print(question['code'], "have not both answers or rows")
#         dict2 = defaultdict(list)
#         def get_response_obj(key, value, scale, root, rank):
#             scale = int(scale) if scale != '' else 0
#             for question_key, list_response in dict2.items():
#                 if key == question_key:
#                     for response in list_response:
#                         if response.value == value and response.scale == scale and response.root == root and response.rank == rank:
#                             return response
#             response = Response(
#                 # code=key,
#                 value=value,
#                 scale=scale,
#                 root=root,
#                 rank=rank
#             )
#             dict2[key].append(response)
#             return response
#         for resp in self.response_json:
#             for q in resp['responseSet']:
#                 question_code = q['questionCode']
#                 for a in q['answerValues']:
#                     if a['answerText'] != '' and a['value']['text'] != '':
#                         question_code = f"{question_code}text{a['answerText']}"
#                         value = a['value']['text']
#                     else:
#                         value = a['answerText'] if a['answerText'] != '' else a['value']['text']
#                     value = str_function.parse_html(value)
#                     scale = a['value']['scale']
#                     rank = int(a['value']['rank']) if 'rank' in a['value'] else 0
#                     root = question_code
#                     loop_on = a.get('sourceAnswerText', None)
#                     if loop_on == '':
#                         loop_on = a['sourceAnswerID']
#                     loop_on = str_function.parse_html(loop_on)
#                     key = f"{question_code}loop{loop_on}"
#                     response_obj = get_response_obj(key, value, scale, root, rank)
#                     response_obj.respondents.append(resp['responseID'])
#         dict3 = defaultdict(lambda: defaultdict(dict))
#         for question_key, list_response in dict2.items():
#             question_code = question_key.split('loop')[0]
#             loop_on = question_key.split('loop')[-1]
#             question_info = dict1.get(question_code.split('text')[0], None)
#             if question_info:
#                 dict3[loop_on][question_code]['info'] = question_info
#                 dict3[loop_on][question_code]['responses'] = list_response
        
#         self.standard_dict = dict3