# Steps to Create and Use .env for API Keys
#   1. pip install python-dotenv
#   2. Create the .env File: In the root directory of your project (where your Python script is),\
<<<<<<< HEAD
#      create a new file called .env. This file will hold your environment variables in the format \n
=======
#      create a new file called .env. This file will hold your environment variables in the format \n 
>>>>>>> vettura/main
#      of KEY=VALUE. Example: OPENAI_API_KEY="your_openai_api_key_here"


from dotenv import load_dotenv
import os
from openai import Client

# Load variables from .env file
load_dotenv()

# Initialize the OpenAI Client
client = Client()

<<<<<<< HEAD

=======
>>>>>>> vettura/main
# Function to query OpenAI
def query(user_input):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": "You are a helpful assistant."},
<<<<<<< HEAD
            {"role": "user", "content": user_input},
=======
            {"role": "user", "content": user_input}
>>>>>>> vettura/main
        ],
        temperature=0.7,
        max_tokens=250,
    )

    return response.choices[0].message.content

<<<<<<< HEAD

=======
>>>>>>> vettura/main
# Main chat loop
def main():
    print("Welcome to OpenAI Chat! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
<<<<<<< HEAD
        if user_input.lower() == "exit":
=======
        if user_input.lower() == 'exit':
>>>>>>> vettura/main
            print("Goodbye!")
            break

        # Generate response
        print("Generating response...")
        response = query(user_input)
        print(f"Assistant: {response}")

<<<<<<< HEAD

=======
>>>>>>> vettura/main
if __name__ == "__main__":
    main()
