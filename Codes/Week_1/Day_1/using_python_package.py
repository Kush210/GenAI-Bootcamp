# Ensure Ollama is installed and running at http://localhost:11434
# To run local ollama server type: "ollama serve" in terminal

import ollama

if __name__ == "__main__":
    totalAvailableQuestions = 3
    currentQues = 0

    while currentQues<totalAvailableQuestions:
        question = str(input("Question: "))
        response = ollama.chat(
            model="llama3.2",
            messages=[{
                "role":"user",
                "content":question
            }],
            stream=True
        )
        for chunk in response:
            print(chunk["message"]["content"],end='',flush=True)
        print('\n')
        currentQues+=1