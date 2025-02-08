# imports
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.chains import create_retrieval_chain
from langchain_openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en", encode_kwargs={"normalize_embeddings": True}
)

# Initialize the vector store
vector_store = Chroma(
    persist_directory="./chromaDB",
    collection_name="job_Descriptions_2",
    embedding_function=embeddings,
)

# Initialize the language model (OpenAI in this case)
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# usage
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# QA chain for combining documents
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

# Initialize the retrieval QA chain
rag_chain = create_retrieval_chain(
    vector_store.as_retriever(search_kwargs={"k": 2}),
    combine_docs_chain,
)

# streamlit interface for job matching QA chatbot

st.title("Job Matching Chatbot")
st.write("Enter your query below to get job recommendations")
query = st.text_input("Enter your query here")
# query = "I am askilled in Java, Rust and Go with 2 years of experience. Looking for a job in North Carolina. Can you provide me some options?"
if query:
    response = rag_chain.invoke({"input": query})
    st.write(response["answer"])
    print(response)
