import os
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

#API Stuff
os.environ['OPENAI_API_KEY'] = ''

#Document Analysis
loader = PyPDFLoader('2020-Scrum-Guide-US.pdf')
loader2 = PyPDFLoader('Handout-Module6.pdf')
page1 = loader.load_and_split()
page2 = loader2.load_and_split()
pages = page1 + page2
#LLM
llm = OpenAI(temperature = 0.9)
embeddings = OpenAIEmbeddings()
#Vector Store Stuff
store = Chroma.from_documents(pages, embeddings ,collection_name = "data")

vectorstore_info = VectorStoreInfo(
    name="data",
    description="Two files, the first one is a guide on scrum, the second one is a guide on making good test questions",
    vectorstore=store
)
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)
prompt = "Can you use the methods in chapter 1 of Is it a Trick Question to generate, a good test question about page 3 of the Scrum Guide??"
agent_executor.run(prompt)
