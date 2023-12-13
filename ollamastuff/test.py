
from langchain.llms import Ollama
from langchain.agents import AgentType, initialize_agent


llm = Ollama(model="mistral")
toolkit = []
agent = initialize_agent(
    toolkit, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose = True)

prompt = "Hello how are you?"
response = agent(prompt)
output = response['output']
print(output)