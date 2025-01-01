import streamlit as st
from unstructured.partition.pdf import partition_pdf
from collections import defaultdict
import re
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Utility Functions
def clean_text(text):
    """Cleans text by removing unwanted characters and excessive whitespace."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,;!?-]', '', text)
    return text.strip()

def process_pdf_with_unstructured(pdf_path):
    """Processes the PDF using unstructured library and organizes data."""
    elements = partition_pdf(
        filename=pdf_path,
        include_page_breaks=True,
        strategy="auto",
        infer_table_structure=True,
        extract_images_in_pdf=True,
    )
    content_by_page = defaultdict(lambda: {"text": [], "images": [], "tables": []})

    for element in elements:
        metadata = element.metadata.to_dict()
        page_number = metadata.get("page_number", "Unknown")
        element_type = getattr(element, "type", None)

        if element_type == "Text":
            content_by_page[page_number]["text"].append(clean_text(element.text or ""))
        elif element_type == "Image":
            content_by_page[page_number]["images"].append(element.to_dict())
        elif element_type == "Table":
            content_by_page[page_number]["tables"].append(element.to_dict())
        else:
            content_by_page[page_number]["text"].append(clean_text(str(element)))

    for page, content in content_by_page.items():
        content["text"] = " ".join(content["text"])

    try:
        del content_by_page['Unknown']
    except Exception as e:
        pass
    return content_by_page

def setup_vectorstore(content_by_page):
    """Creates vector store from processed PDF content."""
    documents = [
        Document(page_content=content["text"], metadata={"page": page})
        for page, content in content_by_page.items()
    ]
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en", encode_kwargs={'normalize_embeddings': True})
    return FAISS.from_documents(documents, embeddings)

def setup_qa_chain(vectorstore):
    """Sets up a retrieval-based QA chain."""
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant. Use the following context to answer the user's question.
Context:
{context}

Question:
{question}

Answer:"""
    )

    llm = ChatOpenAI(api_key="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", model="gpt-3.5-turbo", temperature=0)
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
    if "pdf_content" not in st.session_state:
        st.session_state["pdf_content"] = None
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None
    if "qa_chain" not in st.session_state:
        st.session_state["qa_chain"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = get_chat_history()

# Streamlit App
def main():
    # Initialize session state variables
    initialize_session_state()

    st.title("Query your PDF")
    st.sidebar.header("Upload PDF")
    
    # PDF Upload Section
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
    
    # Submit Button for PDF Processing
    if uploaded_file:
        submit_button = st.sidebar.button("Submit PDF")

        if submit_button:
            with st.spinner("Processing PDF..."):
                pdf_path = uploaded_file.name
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.read())

                st.session_state["pdf_content"] = process_pdf_with_unstructured(pdf_path)
                st.session_state["vectorstore"] = setup_vectorstore(st.session_state["pdf_content"])
                st.session_state["qa_chain"] = setup_qa_chain(st.session_state["vectorstore"])
                st.sidebar.success("PDF processed and indexed!")

    # Chatbot Interface
    st.subheader("Chat with your PDF")
    user_input = st.chat_input("Enter your question:")

    if user_input:
        if not st.session_state.get("qa_chain"):
            st.error("Please upload and process a PDF first.")
        else:
            qa_chain = st.session_state["qa_chain"]
            with st.spinner("Fetching the answer..."):
                result = qa_chain({"query": user_input})
                answer = result["result"]
                sources = result["source_documents"]

                # Append to chat history
                st.session_state["chat_history"].append({"user": user_input, "bot": answer, "sources": sources})
                
    # Display Chat History
    for chat in st.session_state["chat_history"]:
        st.chat_message("user").markdown(chat["user"])
        st.chat_message("assistant").markdown(chat["bot"])

        # Create a container to hold the expanders dynamically
        num_sources = len(chat["sources"])  # Get the number of sources
        num_columns = min(num_sources, 3)  # Limit to 3 columns per row for better display

        # Create columns for the number of sources, limited to 3 per row
        columns = st.columns(num_columns)

        # Loop through the sources and add an expander in each column
        for i, source in enumerate(chat["sources"]):
            # Get the appropriate column index
            col = columns[i % num_columns]

            with col:
                with st.expander(f"Source {i+1} (Page {source.metadata['page']}):"):
                    st.write(f"**Match {i+1}:**")
                    st.write(source.page_content[:300])  # Truncate for brevity


if __name__ == "__main__":
    main()
