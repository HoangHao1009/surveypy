import openai
from typing import List
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnableLambda
import requests
import json

class Answer(BaseModel):
    """Information about answer in a question. Do not care about terminate"""
    text: str = Field(description="Answer content like 16 year old for question: 'how old are you'")
    # question_code: str = Field(description="Question code that answer is in like Q1, Q2, S1, S2.1")

class Question(BaseModel):
    """Information about question."""
    code: str = Field(description="Question code like Q1, Q2, S1, S2.1")
    type: str = Field(description="Question type like ma (multiple answer), sa (single answer), oe (open-end answer)")
    text: str = Field(description="Question content like 'how old are you?'")
    answers: List[Answer] = Field(description="List of answer options")

class Info(BaseModel):
    """Information to extract"""
    questions: List[Question] = Field(description="List of questions")

class Extract_Chain(BaseModel):
    api_key: str
    model: str = "gpt-3.5-turbo"

    def extract(self, pages):
        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=self.api_key)
        extraction_functions = [convert_pydantic_to_openai_function(Info)]
        extraction_model = model.bind(functions=extraction_functions, function_call={"name": "Info"})
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract the relevant information, if not explicitly provided do not guess. Extract partial info"),
            ("human", "{input}")
        ])

        extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="questions")

        prep = RunnableLambda(
            lambda x: [{"input": page.page_content} for page in pages]
        )

        chain = prep | extraction_chain.map()

        return chain.invoke(pages)

class Questionnaire_Creator(BaseModel):
    questionnaire_design_path: str
    extract_chain: Extract_Chain
    api_key: str

    @property
    def pages(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4096,  # Mỗi đoạn có tối đa 50 ký tự
            chunk_overlap=100,  # Các đoạn trùng nhau 10 ký tự
            separators=["New page"]
        )
        loader = PyPDFLoader(self.questionnaire_design_path)
        pages = loader.load_and_split(text_splitter)
    
        return pages

    @property
    def extracted_info(self):
        result = []
        list_info = self.extract_chain.extract(pages=self.pages)
        for info in list_info:
            result.extend(info)
        return result

    @property
    def standard_info(self):
        standard_question_info = []
        extracted_info = self.extracted_info
        for question_info in extracted_info:
            standard_info = question_info
            question_type = None
            if question_info['type'] == 'sa':
                question_type = 'multiplechoice_radio'
            elif question_info['type'] == 'ma':
                question_type = 'multiplechoice_checkbox'
            elif question_info['type'] == 'oe':
                question_type = 'text_single_row'
            standard_info['type'] = question_type
            standard_info['required'] = True
            standard_question_info.append(standard_info)
        return standard_question_info
    
    def create(self, survey_id):
        responses = []
        url = f"https://api.questionpro.com/a/api/v2/surveys/{survey_id}/questions"
        for question in self.standard_info:
            print(question)
            payload = json.dumps(question)
            headers = {
            'Content-Type': 'application/json',
            'api-key': self.api_key
            }
            response = requests.request("POST", url, headers=headers, data=payload)
            responses.append(response)
        return responses
        
        