import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
import streamlit_mermaid
from dotenv import load_dotenv
import os
# Initialize the LLM
load_dotenv()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv("openai_api_key"))

prompt = PromptTemplate(
    template="""
    You are an expert assistant whom user question's to answer questions about technologies by generating mindmap graphs for it.
    Return only the mermaid markdown for the Mindmap graph.Be a bit more detail oriented

    Chat History:{chat_history}
    Question:{input}
    """,
    input_variables=["chat_history", "input"],
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chat_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Streamlit app
st.title("Technology Mindmap Generator")

user_input = st.text_input("Ask a question about technology:")

if user_input:
    # Invoke the LLM Chain with the user's input
    response = chat_chain.invoke({"input": user_input})
    
    # Display the mermaid diagram if there's a response
    if response and response.get("text"):
        st.subheader("Mindmap Graph:")
        res=response["text"].split("\n")
        res=res[1:len(res)-1]
        res_f=""
        for i in res:
            res_f=res_f+i+"\n"
        streamlit_mermaid.st_mermaid(res_f, key="mermaid_diagram")
    else:
        st.write("No response generated.")
