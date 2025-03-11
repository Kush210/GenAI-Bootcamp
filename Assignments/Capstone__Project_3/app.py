from openai import OpenAI
import pandas as pd
import streamlit as st

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import re

load_dotenv()


# Utility Functions
def clean_text(text):
    """Cleans text by removing unwanted characters and excessive whitespace."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,;!?-]", "", text)
    return text.strip()


def setup_vectorstore():
    """Creates vector store from processed PDF content."""
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = Chroma(
        collection_name="Insuarance",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )

    return vector_store


def setup_qa_chain(vectorstore):
    """Sets up a retrieval-based QA chain."""
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 7}
    )

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant. Use the following context to answer the user's question.
Context:
{context}

Question:
{question}

Answer:""",
    )

    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="ft:gpt-4o-mini-2024-07-18:personal::B3RSLUDL",
        temperature=0,
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )


# Cache Messages
@st.cache_data
def get_chat_history():
    """Caches chat history."""
    return []


# Initialize Session State
def initialize_session_state():
    """Initializes required variables in the session state."""
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None
    if "qa_chain" not in st.session_state:
        st.session_state["qa_chain"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = get_chat_history()


def main():
    initialize_session_state()

    st.title("Insuarance customer support Chatbot")

    st.session_state["vectorstore"] = setup_vectorstore()
    st.session_state["qa_chain"] = setup_qa_chain(st.session_state["vectorstore"])

    user_input = st.chat_input("Enter your query")

    if user_input:
        if not st.session_state.get("qa_chain"):
            st.error("Error establishing langchain QA chain.")
        else:
            qa_chain = st.session_state["qa_chain"]
            with st.spinner("Fetching the answer..."):
                result = qa_chain({"query": user_input})
                print(result)
                answer = result["result"]
                sources = result["source_documents"]
                print(sources)

                # Append to chat history
                st.session_state["chat_history"].append(
                    {"user": user_input, "bot": answer, "sources": sources}
                )

    # Display Chat History
    for chat in st.session_state["chat_history"]:
        st.chat_message("user").markdown(chat["user"])
        st.chat_message("assistant").markdown(chat["bot"])

        # Create a container to hold the expanders dynamically
        num_sources = len(chat["sources"])  # Get the number of sources
        num_columns = min(
            num_sources, 3
        )  # Limit to 3 columns per row for better display

        # Create columns for the number of sources, limited to 3 per row
        columns = st.columns(num_columns)

        # Loop through the sources and add an expander in each column
        for i, source in enumerate(chat["sources"]):
            # Get the appropriate column index
            col = columns[i % num_columns]

            with col:
                with st.expander(f"Source {i+1} {source.metadata['source']}"):
                    st.write(f"**Match {i+1}:**")
                    st.write(source.page_content[:300])  # Truncate for brevity


if __name__ == "__main__":
    main()
