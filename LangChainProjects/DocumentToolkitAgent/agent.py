import os
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from document import scrumguide
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.tools.shell.tool import ShellTool
from langchain.tools.python.tool import PythonREPLTool
from apikey import apikey



os.environ['OPENAI_API_KEY'] = apikey
llm = OpenAI(temperature = 0.9)
zool = Tool( 
        name="scrumguide QA system",
        func=scrumguide.run,
        description="useful for when you need to answer questions about scrum. Input should be a fully formed question.",
    )
shool = ShellTool()
tools = [zool,shool,PythonREPLTool()]
agent = initialize_agent(
    tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose = True,handle_parsing_errors=True)
prompt = "What is the point of scrum, why should I use it?"
print(agent.run(prompt))