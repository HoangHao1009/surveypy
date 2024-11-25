from pydantic import BaseModel, Field
import os
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from langchain.schema import Document

from surveypy.core.survey import Survey
from surveypy.core.crosstab import CrossTab
import ast
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.format_scratchpad import format_to_openai_functions

class QuestionCodeInput(BaseModel):
    base: str = Field(description="Base question code (e.g., Q3 for control variables).")
    target: str = Field(description="Target question code (e.g., Q1 for dependent variables).")
    
class Analyzer(BaseModel):
    api_key: str
    survey: Survey
    db_dir: str
    folder_path: str

    def chat(self, query):
        analysis_info = self.analysis_info(query)
        analysis_request = self.llm.invoke(f"Synthesis this question to a request for input in further step: {query}. You can also you below information if there is relavant:\n{analysis_info}")
        answer = self.ctab_agent.invoke(
            {"input": f"{analysis_request['output']}"},
        )
        return answer

    @property
    def llm(self):
        return ChatOpenAI(openai_api_key=self.api_key, temperature=0)
    
    @property
    def ctab_agent(self):
        @tool(args_schema=QuestionCodeInput)
        def ctab_data(base: str, target: str) -> str:
            """Get crosstab data between base question and target question. Use when analyzing relationship between variables"""

            base_question_list = [self.survey[base]]
            target_question_list = [self.survey[target]]
            ctab = CrossTab(
                bases=base_question_list,
                targets=target_question_list,   
            )
            ctab_data = ctab.dataframe.to_string()
            return f"Here is crosstab data between {base} and targets {target}: {ctab_data}"

        tools = [ctab_data]
        llm = self.llm

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are survey analyzer who professional in market research."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        functions = [format_tool_to_openai_function(f) for f in tools]
        model = llm.bind(functions=functions)

        agent_chain = RunnablePassthrough.assign(
            agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
        ) | prompt | model | OpenAIFunctionsAgentOutputParser()

        agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True)
        return agent_executor
    
    def analysis_info(self, analysis_query):
        retriever = self.db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'k': 3, 'score_threshold': 0.1}
        )
        relevant_info_docs = retriever.invoke(analysis_query)
        
        relevant_info_text = analysis_query + '\n'.join([i.page_content for i in relevant_info_docs])
                        
        return relevant_info_text
        
    @property
    def emb_model(self):
        return OpenAIEmbeddings(api_key=self.api_key)
    
    @property
    def survey_info(self):
        all_info = []
        for question in self.survey.questions:
            info = f"Here is question code {question.code} with\n"
            info += f"Question text is '{question.text}'\n"
            info += f"Question type is '{question.type}'\n"
            info += f"Question answer is:\n"
            for response in question.responses:
                info += f"{response.scale} - {response.value}\n"
            info += '------------------------------------------'
            all_info.append(info)
        return all_info
            
    @property
    def db(self):
        all_docs = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4096, 
            chunk_overlap=100,
        )

        files = os.listdir(self.folder_path)
        survey_info_documents = [Document(page_content=text) for text in self.survey_info]

        if len(files) > 0:
            for filename in files:
                if filename.endswith(".pdf"):
                    pdf_file_path = os.path.join(self.folder_path, filename)
                    loader = PyPDFLoader(pdf_file_path)
                    docs = loader.load_and_split(text_splitter)
                    all_docs.extend(docs)
            
            vector_db = Chroma.from_documents(all_docs, self.emb_model, persist_directory=self.db_dir)
            vector_db.add_documents(survey_info_documents)
        else:
            vector_db = Chroma.from_documents(survey_info_documents, self.emb_model, persist_directory=self.db_dir)        
        return vector_db        
