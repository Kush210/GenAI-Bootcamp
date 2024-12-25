import requests
import json
# TODO: Add instruction video link to install Ollama locally

# Ensure Ollama is installed and running at http://localhost:11434
# To run local Ollama server type: "ollama serve" in terminal
# Refer to: https://github.com/ollama/ollama/blob/main/docs/api.md for more details

if __name__ == "__main__":
    totalAvailableQues = 3  # Number of questions the user can ask
    currentQues = 0  # Starting question index

    while currentQues < totalAvailableQues:
        # Taking input from the user
        que = str(input("Question: "))  

        # Ollama API URL
        url = "http://localhost:11434/api/chat"

        # Preparing the payload to send to Ollama
        payload = {
            "model": "llama3.2",  # Specify the model you're using (adjust as necessary)
            "messages": [
                {"role": "user", "content": que}  # User's question is sent as content
            ],
            "stream": False  # Change this to True for streaming responses
        }

        if payload["stream"]:
            # for streaming results
            try:
                with requests.post(url, json=payload, stream=True) as response:
                    if response.status_code == 200:
                       
                        for chunk in response.iter_content(chunk_size=None):
                            if chunk: 
                                try:
                                    chunk_decoded = json.loads(chunk.decode("utf-8"))
                                    if "message" in chunk_decoded and "content" in chunk_decoded["message"]:
                                        print(chunk_decoded["message"]["content"], end="", flush=True)
                                except json.JSONDecodeError:
                                    continue
                        print()  
                        currentQues += 1
                    else:
                        print(f"Error: {response.status_code} - {response.reason}")
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            # For non streaming results
            try:
                response = requests.post(url, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    print("Response:", data["message"]["content"])
                    currentQues += 1
                else:
                    print(f"Error: {response.status_code} - {response.reason}")
            except Exception as e:
                print(f"An error occurred: {e}")
