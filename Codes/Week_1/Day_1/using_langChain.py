# Ensure Ollama is installed and running at http://localhost:11434
# To run local ollama server type: "ollama serve" in terminal

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

if __name__ == "__main__":
    totalAvailableQues = 3
    currentQues = 0

    template = """Question:{question}

    Answer: """

    stream = False # for streaming the response

    model = ChatOllama(model="llama3.2")
    prompt = ChatPromptTemplate.from_template(template=template)

    chain = prompt | model

    while currentQues < totalAvailableQues:
        que = str(input("Question: "))
        if stream:
            response = chain.stream({"question": que})
            for chunk in response:
                print(chunk.content,end='',flush=True)
            print('\n')
            currentQues+=1
        else:
            response = chain.invoke({"question": que})
            print(response.content, end="", flush=True)  
            print() 
            currentQues += 1

