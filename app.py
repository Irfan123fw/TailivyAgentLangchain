import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.llm import LLMChain
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.document_transformers import LongContextReorder
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
)
import openai
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains.llm import LLMChain
import os
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import os
from langchain import PromptTemplate
from dotenv import load_dotenv
from streamlit_chat import message
load_dotenv()
import openai
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import ConversationalChatAgent, AgentExecutor, create_json_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain.tools.retriever import create_retriever_tool
from langchain_community.chat_models import ChatOpenAI
from langchain import hub
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.retrievers.tavily_search_api import TavilySearchAPIRetriever
from langchain.tools.tavily_search import TavilySearchResults

retriever = TavilySearchAPIRetriever(k=20, include_domains = ["Your_Domain"])
# tavily_tool = TavilySearchResults(api_wrapper=retriever, max_results=20) include_raw_content = True
st.set_page_config(page_title="Chatbot", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title(" Question Answering ")
conversation_history = []
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
def serialize_history(request):
        converted_chat_history = []
        for message in request:
            if message.get("human") is not None:
                converted_chat_history.append(HumanMessage(content=message["human"]))
            if message.get("ai") is not None:
                converted_chat_history.append(AIMessage(content=message["ai"]))
        return converted_chat_history
    
Qaprompt = hub.pull("hwchase17/react-chat-json")
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)
if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state.steps = {}

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)

if prompt := st.chat_input(placeholder="bagaimana keindahan surga?"):
    st.chat_message("user").write(prompt)
    conversation_history.append({"human": f"{prompt}"})
    historys = serialize_history(conversation_history)
    tool = create_retriever_tool(
        retriever,
        "Searching ",
        "Searches and returns document .If there is nothing in the context relevant to the question at hand, just say you dont know Dont try to make up an answer.always answer using indonesian language. You must use this tool!",
    )
    llm = ChatOpenAI( model_name= "gpt-3.5-turbo-0125",streaming=True, temperature=0, max_tokens=3000)
    tools = [tool]
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
    # agent = create_json_chat_agent(llm, tools)
    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    c=retriever.invoke(prompt)
    print(c)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        cfg = RunnableConfig()
        cfg["callbacks"] = [st_cb]
        response = executor.invoke(prompt, cfg)
        st.write(response["output"])
        answer = response["output"]
        conversation_history.append({"ai": f"{answer}"})
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]