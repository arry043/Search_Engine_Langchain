import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent, AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv


# Arxiv and Wikipedia Weapper
api_wraper_wiki=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=api_wraper_wiki, name="Wikipedia")

api_wraper_arxiv=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=api_wraper_arxiv, name="Arxiv")

search=DuckDuckGoSearchRun(name="Search")



st.title("Search Engine - GROQ & Ollama")


"""
StreamlitCallbackHandler is a callback handler that prints to streamlit. 
"""



# Sidebar
st.sidebar.title("Search Engine Settings")
groq_api_key=st.sidebar.text_input("GROQ API Key", type="password")


if "messages" not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", "content": "Hello! How can I help you?"}]
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

    
if prompt:=st.chat_input(placeholder="What is Machine Learning?"):
    if not groq_api_key:
        st.warning("Please enter your OpenAI API Key to continue.")
        st.stop()  
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    llm=ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant", streaming=True)
    tools=[wiki, arxiv, search]
    
    # search_agent=initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, max_iterations=3, handle_parsing_errors=True)
    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        # verbose=True,
        max_iterations=7,
        handle_parsing_errors=True
    )

    
    with st.chat_message('assistant'):
        st_callback=StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        # response=search_agent.run(st.session_state.messages, callbacks=[st_callback])
        response = search_agent.run(prompt, callbacks=[st_callback])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)