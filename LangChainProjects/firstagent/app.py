import os 

import streamlit as st
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory


os.environ['HUGGINGFACEHUB_API_TOKEN'] = ''

st.title('Video Game Title Maker Guy')
prompt = st.text_input('give me a video game title about...')
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'give me a video game title about {topic}'
)
game_template = PromptTemplate(
    input_variables = ['title'],
    template = 'describe the gameplay of a game based on this title Title: {title}'
)

memory = ConversationBufferMemory(input_key = 'topic', memory_key = 'chat_history')

repo_id = "google/flan-t5-xxl"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.9, "max_length": 100})
title_chain = LLMChain(llm=llm, prompt = title_template, verbose=True,memory=memory)
game_chain = LLMChain(llm=llm,prompt = game_template, verbose = True)
sequential_chain = SimpleSequentialChain(chains=[title_chain, game_chain], verbose = True)
if prompt:
    response = sequential_chain.run(prompt)
    st.write(response)
    with st.expander('Message History'):
        st.info(memory.buffer)