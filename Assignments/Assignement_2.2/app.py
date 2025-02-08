import streamlit as st
import chromadb
from chromadb.config import Settings
from matplotlib import pyplot as plt
from PIL import Image
import base64
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import torch

if "query_text" not in st.session_state:
    st.session_state["query_text"] = None
if "uploaded_image" not in st.session_state:
    st.session_state["uploaded_image"] = None
if "result_image" not in st.session_state:
    st.session_state["result_image"] = None

# Initialize Chroma client to load the data from the disk
client = chromadb.PersistentClient(
    path="./chromaDB1", settings=Settings(allow_reset=False)
)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = processor.tokenizer


# Function to get image embeddings
def get_image_embeddings(image):
    inputs = processor(images=image, return_tensors="pt")
    # Get embeddings for the image
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    # Detach from the computational graph, then normalize the embeddings
    outputs = outputs.detach()
    outputs = outputs.squeeze().tolist()
    return outputs


# Function to get text embeddings
def get_text_embeddings(text):
    inputs = processor(text=[text], return_tensors="pt")
    outputs = model.get_text_features(**inputs)
    outputs = outputs.detach()
    outputs = outputs.squeeze().tolist()
    return outputs


def image_to_base64(image):
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format="JPEG")  # You can change the format (e.g., 'PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode("utf-8")


# Load the collection from ChromaDB (ensure the collection name is the same as when it was created)
collection = client.get_collection("image_embeddings")


st.title("Image Search Engine")
st.write("Search for images using text or upload an image")

query_text = st.text_input("Enter a text description or upload an image")
if query_text:
    st.session_state["query_text"] = query_text

# Create a file uploader for the image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    st.session_state["uploaded_image"] = uploaded_image

if st.session_state["uploaded_image"] is not None:
    # Open the uploaded image
    uploaded_image = st.session_state["uploaded_image"]
    st.write("Searching for similar images...")
    image = Image.open(uploaded_image)
    image_embedding = get_image_embeddings(image)
    st.session_state["result_image"] = collection.query(
        query_embeddings=image_embedding, n_results=3
    )["documents"][0]
    st.session_state["uploaded_image"] = None


if st.session_state["query_text"] is not None:
    query_text = st.session_state["query_text"]
    st.write("Searching for similar images...")
    print("Query text", query_text)
    query_embedding = get_text_embeddings(query_text)

    st.session_state["result_image"] = collection.query(
        query_embeddings=query_embedding, n_results=3
    )["documents"][0]

    st.session_state["query_text"] = None

if st.session_state["result_image"] is not None:
    image_base64 = st.session_state["result_image"]
    for image in image_base64:
        image_data = base64.b64decode(image)
        image = Image.open(BytesIO(image_data))
        st.image(image, use_container_width=True)
    st.session_state["result_image"] = None
