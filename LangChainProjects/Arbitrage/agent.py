import os
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, Tool
from apikey import apikey
from basketapi import basketapi
import requests
from tools import bballtool
os.environ['OPENAI_API_KEY'] = apikey

llm = OpenAI(temperature = 0.9)
zool = Tool( 
        name="API-Basketball",
        func=basketapi.run,
        description="Use when you need to use the basketball api documentation, ask questions only",
    )


tools = [zool,bballtool]
agent = initialize_agent(
    tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose = True,handle_parsing_errors=True)
prompt = "Use bballtool to get the head to heads of the teams"
response = agent.run(prompt)
print(response)