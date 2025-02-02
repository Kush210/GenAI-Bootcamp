import streamlit as st
from unstructured.partition.pdf import partition_pdf
from collections import defaultdict
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import re
import os
import fitz  
import base64
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

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

    content_by_page.pop('Unknown', None)
    return content_by_page

def pdf_to_images_with_base64(pdf_path):
    """Converts PDF pages to base64-encoded images."""
    pdf_document = fitz.open(pdf_path)
    page_metadata = defaultdict(dict)

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        image_bytes = pix.tobytes(output="png")
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        page_metadata[page_num + 1] = {"base64_image": base64_image}

    pdf_document.close()
    return page_metadata

def setup_vectorstore(content_by_page, pdf_path):
    """Creates vector store from processed PDF content."""
    page_images = pdf_to_images_with_base64(pdf_path)
    documents = [
        Document(page_content=content["text"], metadata={"page": page, "page_image": page_images[page]})
        for page, content in content_by_page.items()
    ]
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en", encode_kwargs={'normalize_embeddings': True})
    return FAISS.from_documents(documents, embeddings)

def send_images_to_openai_vision(images, query):
    """Sends multiple base64-encoded images to OpenAI Vision model along with the query."""
    content = [{"type": "text", "text": query}]
    for image in images:
        image_data = f"data:image/png;base64,{image}"
        content.append({"type": "image_url", "image_url": {"url": image_data}})

    response = st.session_state.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": content}]
    )
    # print(response)
    return response

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
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = get_chat_history()
    if "client" not in st.session_state:
        st.session_state.client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

# Streamlit App
def main():
    initialize_session_state()
    st.title("Vision-Powered PDF RAG")
    st.sidebar.header("Upload PDF")
    
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        submit_button = st.sidebar.button("Submit PDF")

        if submit_button:
            with st.spinner("Processing PDF..."):
                pdf_path = uploaded_file.name
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.read())

                st.session_state["pdf_content"] = process_pdf_with_unstructured(pdf_path)
                st.session_state["vectorstore"] = setup_vectorstore(st.session_state["pdf_content"], pdf_path)
                st.sidebar.success("PDF processed and indexed!")

    # Chatbot Interface
    user_input = st.chat_input("Enter your question:")

    if user_input:
        if not st.session_state.get("vectorstore"):
            st.error("Please upload and process a PDF first.")
        else:
            retriever = st.session_state["vectorstore"].as_retriever(search_type="similarity", search_kwargs={"k": 3})
            with st.spinner("Fetching the answer..."):
                results = retriever.get_relevant_documents(user_input)
                sources = []
                for doc in results:
                    page_number = doc.metadata.get("page", None)
                    page_content = doc.page_content
                    
                    # print("Page Number: ", page_number)
                    # print("Page Content: ", page_content)
                    if page_number is not None and page_content is not None:
                        sources.append({'page': page_number, 'content': page_content})

                images = [doc.metadata["page_image"]["base64_image"] for doc in results]

                response = send_images_to_openai_vision(images, user_input)
                answer = response.choices[0].message.content
                # print("Sources: ", sources)
                # Append to chat history
                st.session_state["chat_history"].append({"user": user_input, "bot": answer, "sources": sources})

    # Display Chat History
    for chat in st.session_state["chat_history"]:
        st.chat_message("user").markdown(chat["user"])
        st.chat_message("assistant").markdown(chat["bot"])
        num_sources = len(chat["sources"])  # Get the number of sources
        # print(num_sources)
        num_columns = min(num_sources, 3)  # Limit to 3 columns per row for better display

        # Create columns for the number of sources, limited to 3 per row
        columns = st.columns(num_columns)

        # Loop through the sources and add an expander in each column
        for i, source in enumerate(chat["sources"]):
            # Get the appropriate column index
            col = columns[i % num_columns]

            with col:
                with st.expander(f"Source {i+1} (Page {source['page']}):"):
                    st.write(f"**Match {i+1}:**")
                    st.write(source['content'][:300]) 

if __name__ == "__main__":
    main()
