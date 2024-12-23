import requests

#TODO: Add instruction video link to install ollama locally
#TODO: Streaming = True is not working

# Ensure Ollama is installed and running at http://localhost:11434
# To run local ollama server type: "ollama serve" in terminal
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
            "stream": False  # Set to True if you want streaming responses
        }

        # Send POST request to Ollama API
        response = requests.post(url, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            print(data["message"]["content"]) 
            currentQues += 1 
        else:
            print(f"Error: {response.status_code}")  # Print error if the request fails
