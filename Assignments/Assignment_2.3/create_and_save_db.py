# imports
from tqdm import tqdm
from datasets import load_dataset
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain

import chromadb
from chromadb.config import Settings

# Load this image dataset from hugging face -> frgfm/imagenette
dataset = load_dataset("cnamuangtoun/resume-job-description-fit", split="train")[
    "job_description_text"
]

unique_values = [set(dataset)]

# get unique values in the dataset
unique_values = set(dataset)
unique_dataset = []
for element in unique_values:
    unique_dataset.append(element)
print(len(unique_dataset))

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en", encode_kwargs={"normalize_embeddings": True}
)

# Create a new client for chromaDB
client = chromadb.PersistentClient(
    path="./chromaDB", settings=Settings(allow_reset=True)
)
client.reset()

# Create a new collection called 'bbc_news'
job_collection = client.create_collection(name="job_Descriptions_2")

for i in tqdm(range(len(unique_dataset)), desc="loading job descriptions"):
    embedding = embeddings.embed_query(unique_dataset[i])
    job_collection.add(
        ids=[str(i)],  # Unique ID for each job description
        documents=[unique_dataset[i]],  # The job description data (document)
        embeddings=[embedding],  # The corresponding embedding
        metadatas=[
            {
                "id": i,
            }
        ],  #  metadata
    )
