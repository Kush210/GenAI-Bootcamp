from huggingface_hub import InferenceClient

## Define and use your api key/ Access Token
client = InferenceClient(api_key="")

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