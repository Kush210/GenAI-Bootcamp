from huggingface_hub import InferenceClient
from huggingface_hub import login
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Get the token from .env
token = os.getenv("HUGGINGFACE_TOKEN")

if token:
    # Login to Hugging Face
    login(token=token)
    print("Successfully logged in to Hugging Face.")
else:
    print("Token not found. Please check your .env file.")
    
## Define and use your api key/ Access Token
client = InferenceClient(api_key=token)

# Set the role and messages

messages = [
	{
		"role": "user",
		"content": "What is the capital of France?"
	}
]

# define the completions 
stream = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3", 
	messages=messages, 
	max_tokens=500,
	stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content, end="")
    