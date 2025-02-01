# Steps to Create and Use .env for API Keys
#   1. pip install python-dotenv
#   2. Create the .env File: In the root directory of your project (where your Python script is),\
#      create a new file called .env. This file will hold your environment variables in the format \n
#      of KEY=VALUE. Example: OPENAI_API_KEY="your_openai_api_key_here"


from dotenv import load_dotenv
import os
from openai import Client

# Load variables from .env file
load_dotenv()

# Initialize the OpenAI Client
client = Client()


# Function to query OpenAI
def query(user_input):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input},
        ],
        temperature=0.7,
        max_tokens=250,
    )

    return response.choices[0].message.content


# Main chat loop
def main():
    print("Welcome to OpenAI Chat! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Generate response
        print("Generating response...")
        response = query(user_input)
        print(f"Assistant: {response}")


if __name__ == "__main__":
    main()
