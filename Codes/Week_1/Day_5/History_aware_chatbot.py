import streamlit as st
import requests as re

<<<<<<< HEAD

# Function to send query to the API
@st.cache_data
def query(user_input, context):
    url = "http://localhost:11434/api/chat"

    system = {
        "role": "system",
        "content": "Your are Vettura, an obedient and helpful AI assistant. Help the user with his questions and be nice to the user.",
    }

    messages = []
=======
# Function to send query to the API
@st.cache_data
def query(user_input, context):
    url = 'http://localhost:11434/api/chat'

    system = {
        "role" : "system",
        "content" : "Your are Vettura, an obedient and helpful AI assistant. Help the user with his questions and be nice to the user."
    }
    
    messages  = []
>>>>>>> vettura/main

    messages.append(system)
    for msg in context:
        messages.append(msg)
<<<<<<< HEAD

    user = {"role": "user", "content": user_input}

=======
    
    user = {
        "role" : "user",
        "content" : user_input
    }
    
>>>>>>> vettura/main
    messages.append(user)

    st.write(messages)

    payload = {
        "model": "gemma:2b",
        "messages": messages,
        "stream": False,
<<<<<<< HEAD
        "options": {"temperature": 0.5},
=======
        "options" : {
            "temperature" : 0.5
        }
>>>>>>> vettura/main
    }

    response = re.post(url=url, json=payload)
    return response.json()  # Return the full response to include detailed explanations


# Initialize the chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Vettura Chat")

# Roles explained:
# "system": Represents system-level instructions or configurations for the bot.
# "user": Represents the user's messages and queries in the conversation.
# "assistant": Represents the bot's responses to the user's queries.

<<<<<<< HEAD

=======
>>>>>>> vettura/main
# Function to generate a response and maintain conversation history
def generate_response(user_input):
    # Include the last 10 messages as context for the conversation
    context = st.session_state.chat_history[-5:]  # Take the last 5 messages only
    # context
    # Query the API with the user's input and context
    response = query(user_input, context)

    # Append the user's input to the chat history
<<<<<<< HEAD
    st.session_state.chat_history.append({"role": "user", "content": user_input})
=======
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_input
        }
    )
>>>>>>> vettura/main

    # Append the bot's response to the chat history
    st.session_state.chat_history.append(
        {
            "role": "assistant",
<<<<<<< HEAD
            "content": response["message"][
                "content"
            ],  # Assuming API response contains a 'message' key
=======
            "content": response['message']['content']  # Assuming API response contains a 'message' key
>>>>>>> vettura/main
        }
    )

    # Display the conversation on the UI
    for message in st.session_state.chat_history:
<<<<<<< HEAD
        with st.chat_message(message["role"]):
            print("printin message...")
            st.markdown(message["content"])

    # st.session_state.chat_history


# Input box for user message as
=======
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    
    # st.session_state.chat_history

# Input box for user message
>>>>>>> vettura/main
if user_input := st.chat_input("Your Message"):
    generate_response(user_input)
