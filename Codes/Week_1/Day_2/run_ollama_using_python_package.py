# Ensure Ollama is installed and running at http://localhost:11434
# To run local ollama server type: "ollama serve" in terminal

import ollama

if __name__ == "__main__":
    totalAvailableQuestions = 3
    currentQues = 0
    streaming_Mode = False

    while currentQues<totalAvailableQuestions:
        question = str(input("Question: "))
        response = ollama.chat(
            model="gemma:2b", #"llama3.2", 
            messages=[{
                "role":"user",
                "content":question
            }],
            stream=streaming_Mode
        )

        if(streaming_Mode):
            for chunk in response:
                print(chunk["message"]["content"],end='',flush=True)
            print('\n')
        else:
            print(response["message"]["content"])
        currentQues+=1