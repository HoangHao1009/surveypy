from pydantic import BaseModel, Field
import os
from typing import List, ClassVar

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool

from ..core.survey import Survey
from ..core.crosstab import CrossTab

class QuestionCodeInput(BaseModel):
    bases: List[str] = Field(description="List of base questions code with base mean control variables")
    targerts: List[str] = Field(description="List of target questions code with target mean target variables")

class Analyzer(BaseModel):
    api_key: str
    survey: Survey
    db_dir: str
    folder_path: str
    ctab_data: ClassVar = tool(args_schema=QuestionCodeInput)(
        lambda self, bases, targets: self._ctab_data(bases, targets)
    )
    
    def _ctab_data(self, bases: List[str], targets: List[str]):
        """Get crosstab data between base questions and target questions"""
        base_questions = [self.survey[question_code] for question_code in bases]
        target_questions = [self.survey[question_code] for question_code in targets]
        ctab = CrossTab(
            bases=base_questions,
            targets=target_questions,   
        )
        ctab_data = ctab.dataframe.to_string()
        return f"Here is crosstab data between {bases} and targets {targets}: {ctab_data}"
    
    @property
    def ctab_agent(self):
        tools = [self.ctab_data]
        prompt = hub.pull('hwchase17/react')
        llm = ChatOpenAI(openai_api_key=self.api_key, temperature=0)
        
        agent = create_react_agent(
            llm=llm,
            prompt=prompt,
            tools=tools,
            stop_sequence=True
        )
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True
        )
        
        return agent_executor
    
    def analysis_info(self, analysis_query):
        retriever = self.db.as_retriever(
            search_type="similarity_score_threshhold",
            search_kwargs={'k': 3, 'score_threshhold': 0.1}
        )
        question_info = retriever.invoke(analysis_query)
        
        question_info_text = '\n'.join([i.page_content for i in question_info])
        
        ctab_data = self.ctab_agent.invoke(
            {"input": f"From these information, take bases question code and targets question code for taking crosstab data: {question_info_text}"}
        )
        
        external_query = f"Return relavent information for analysis: '{analysis_query} for question: {question_info}'"
        
        external_info = retriever.invoke(external_query)
        
        info = []
        for i in question_info + external_info:
            info += f"Document: {i.page_content} Source: {i.metadata['source']}\n"
        
        info += ctab_data
        return info
        
    @property
    def emb_model(self):
        return OpenAIEmbeddings(api_key=self.api_key)
    
    @property
    def survey_info(self):
        info = ''
        for question in self.survey.questions:
            info = f"Here is question code {question.code} with\n"
            info += f"Question text is '{question.text}'\n"
            info += f"Question type is '{question.type}'\n"
            info += f"Question answer is:\n"
            for response in question.responses:
                info += f"{response.scale} - {response.value}\n"
            info += '------------------------------------------'
        return info
            
    @property
    def db(self):
        if not os.path.exists(self.db_dir):
            all_docs = []
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, 
                chunk_overlap=100,
            )

            for filename in os.listdir(self.folder_path):
                if filename.endswith(".pdf"):
                    pdf_file_path = os.path.join(self.folder_path, filename)
                    loader = PyPDFLoader(pdf_file_path)
                    docs = loader.load_and_split(text_splitter)
                    all_docs.extend(docs)
                
            vector_db = Chroma.from_documents(all_docs, self.emb_model, persist_directory=self.db_dir)
            survey_info_docs = text_splitter.split_text(self.survey_info)
            vector_db.add_documents(survey_info_docs)
            
            return vector_db
        else:
            print('Db existed')
        
