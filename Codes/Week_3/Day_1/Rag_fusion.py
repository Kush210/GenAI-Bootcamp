import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from warnings import filterwarnings
from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads

filterwarnings('ignore')
load_dotenv()


def setup_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en", encode_kwargs={'normalize_embeddings': True})
    return FAISS.from_documents(chunks, embeddings)

# Reciprocal Rank Fusion algorithm
def reciprocal_rank_fusion(search_results_dict, k=60):
    fused_scores = {}
    for docs in search_results_dict:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            # print('\n')
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

        # final reranked result
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        return reranked_results

# Function to process text file into chunks and extract propositions
def process_text_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size =400, chunk_overlap=60) # ["\n\n", "\n", " ", ""] 65,450
    chunks = text_splitter.create_documents([text])
    return chunks  

# Main function
if __name__ == "__main__":
    # Read text file and process into chunks
    text_file_path = r"D:\Projs\Epilepto\kangra_fort.txt"  # Replace with the path to your text file
    chunks = process_text_file(text_file_path)

    # print(chunks)
    vectorstore = setup_vectorstore(chunks)
    retriever = vectorstore.as_retriever()

    template = """You are a helpful assistant tasked with generating multiple search queries focused on tourism aspects of Kangra Fort. 
    Create four search queries related to: {question} 
    Output (4 tourism-related queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)


    generate_queries = (
        prompt_rag_fusion 
        | ChatGroq(temperature=0,model="llama-3.3-70b-versatile")
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
    )

    retrieval_chain_rag_fusion = generate_queries | retriever.map()
    # original question
    original_question = 'A fort in Kangra ?'
    
    # retrieve documents
    print("queries",retrieval_chain_rag_fusion)
    results = retrieval_chain_rag_fusion.invoke({"question": original_question})
    print("generated queries :", results)

    print(len(results))
    # we have 4 generated questions

    print(len(results[0]))

    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # llm = ChatOpenAI(temperature=0)
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=None,
        timeout=None,
    )

    final_rag_chain = (prompt
        | llm
        | StrOutputParser()
    )

    ans = final_rag_chain.invoke({"context":reciprocal_rank_fusion(results),"question":original_question})
    print(ans)