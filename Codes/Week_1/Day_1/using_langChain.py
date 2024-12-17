# Ensure Ollama is installed and running at http://localhost:11434
# To run local ollama server type: "ollama serve" in terminal

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

if __name__ =="__main__":
    totalAvailableQues = 3
    currentQues = 0

    template = """Question:{question}
    
    Answer: """

    model = ChatOllama(model="llama3.2")
    prompt = ChatPromptTemplate.from_template(template=template)

    chain = prompt | model

    while currentQues<totalAvailableQues:
        que = str(input("Question: "))
        response = chain.invoke({"question":que})
        print(response.content)
        currentQues+=1
