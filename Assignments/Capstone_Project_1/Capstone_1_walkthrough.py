import asyncio
import streamlit as st
import os
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv
import ollama
import json

# Load environment variables
load_dotenv()

# initialize clients
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
ollama_client = ollama.Client()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "counter" not in st.session_state:
    st.session_state.counter = 0


# Function to handle Steve's response
@st.cache_data
def query_steve(user_input, context):
    """
    This function should:
    - Set up a system prompt for Steve Jobs' persona and debating style.
    - Pass the context and user input to an API (e.g., Anthropic Claude or OpenAI).
    - Return Steve's response as a concise message (max 200 words).
    """
    print("querying steve ...")

    # reduced context to last 3 messages to save on rate limits
    context = st.session_state.chat_history[-3:]
    system = {
        "role": "system",
        "content": """You are Steve Jobs, the co-founder of Apple. You are known for your visionary thinking
         and persuasive communication style. Your responses should reflect your innovative mindset and ability
         to inspire others. Keep your responses concise and impactful.(max 200 words). You are debating with Elon Musk. so include a relevant question in your response for Elon Musk""",
    }

    messages = []
    messages.append(system)
    for msg in context:
        messages.append(msg)
    user = {"role": "user", "content": user_input}
    messages.append(user)

    # call open ai for response
    response = openai.chat.completions.create(
        model="gpt-4o", messages=messages, temperature=0.5
    )
    return response.choices[
        0
    ].message.content  # Return the full response to include detailed explanations


# Function to handle Elon's response
@st.cache_data
def query_elon(user_input, context):
    """
    This function should:
    - Set up a system prompt for Elon Musk's persona and debating style.
    - Pass the context and user input to an API (e.g., OpenAI GPT-4 or Groq API).
    - Return Elon's response as a concise message (max 200 words).
    """
    print("querying Elon ...")
    # reduced context to last 3 messages to save on rate limits
    context = st.session_state.chat_history[-3:]
    system = {
        "role": "system",
        "content": """You are Elon Musk, the CEO of Tesla and SpaceX. You are known for your ambitious
          goals and ability to think outside the box. Your responses should reflect your forward-thinking 
          approach and willingness to take risks. Keep your responses concise and engaging.(max 200 words)You are debating with Steve Jobs. So include a relevant question in your response for Steve Jobs""",
    }
    messages = []
    messages.append(system)
    for msg in context:
        messages.append(msg)
    user = {"role": "user", "content": user_input}
    messages.append(user)

    # response = ollama_client.chat(
    #     model="llama3.3-70b", messages=messages, temperature=0.5
    # )

    response = groq.chat.completions.create(
        model="llama-3.1-8b-instant", messages=messages, temperature=0.5
    )

    return response.choices[
        0
    ].message.content  # Return the full response to include detailed explanations


# Function to simulate the debate
def simulate_debate(user_input):
    """
    This function should:
    - Alternate between querying Steve and Elon for a set number of turns (e.g., 50).
    - Maintain a conversation history in `st.session_state.chat_history`.
    - Use each bot's previous response to generate the next input for the other.
    """
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Let the debate happen for a sequence of 50 conversations.
    # Due to repeated expirations of rate limits the number of conversations is reduced to 15
    while st.session_state.counter < 50:
        st.session_state.counter += 1
        print("Conversation counter", st.session_state.counter)

        # Query Steve and Elon alternately
        input_1 = st.session_state.chat_history[-1]["content"]
        response_1 = query_steve(input_1, st.session_state.chat_history)

        response_2 = query_elon(response_1, st.session_state.chat_history)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response_1}
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": response_2}
        )

    # Display the conversation on the UI
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# Function for judging the debate
def judge():
    """
    This function should:
    - Analyze the entire debate stored in `st.session_state.chat_history`.
    - Use a language model to evaluate the debate on criteria like clarity, engagement, and tone.
    - Determine and return the winner.
    """
    votes = {"assistant_Steve_Jobs": 0, "assistant_Elon_Musk": 0}
    summary = ""

    print("Judging the debate ...")

    # uses batach of 4 conversation to vote a winner and provide summary
    context = st.session_state.chat_history[1:9]
    system_prompt = """You are a debate judge. evaluate the debate on criteria like clarity, engagement, and tone. 
    You have to analyze the debate between assistant_Steve_Jobs and assistant_Elon_Musk. 
    Just give the response with the assistant name who won the debate. Do not give any explanation.\n
    Just give the name of the assistant who won the debate.\n
    Reply in json format containing the winner and summary on the reasons why you chose that winner.\n
    \n
    Output format:\n
    {"winner": "assistant_Steve_Jobs or assistant_Elon_Musk",\n
     "summary": "reason for winning the debateevaluate the debate on criteria like clarity, engagement, and tone"}\n
    """

    system_prompt += f"debate_ topic: {st.session_state.chat_history[0]['content']}\n"

    for i in range(1, len(st.session_state.chat_history), 6):
        context = st.session_state.chat_history[i : i + 6]
        assistant_responses = [
            entry["content"] for entry in context if entry["role"] == "assistant"
        ]

        for idx, response in enumerate(assistant_responses):
            if idx % 2 == 0:
                system_prompt + f"assistant_Steve_Jobs: {response}\n"
            else:
                system_prompt + f"assistant_Elon_Musk: {response}\n"

        response = groq.chat.completions.create(
            model="llama-3.3-70b-specdec",
            messages=[{"role": "user", "content": system_prompt}],
            temperature=0.5,
        )

        results = response.choices[0].message.content
        # convert results to json

        results = json.loads(results)
        winner = results["winner"]
        summary = summary + results["summary"] + "\n"
        votes[winner] += 1

    winner = max(votes, key=votes.get)
    final_summary = groq.chat.completions.create(
        model="llama-3.3-70b-specdec",
        messages=[
            {
                "role": "user",
                "content": f"Summarize the reasons why {winner} won the debate.Use the following summary:\n{summary}",
            }
        ],
        temperature=0.5,
    )
    print(votes, winner, summary)

    return (winner, final_summary.choices[0].message.content)


# Streamlit UI setup
def main():
    """
    Main Streamlit app logic:
    - Initialize the debate and display responses as they happen.
    - Provide an option to see the judge's verdict at the end.
    """
    st.title("Steve vs. Elon: The Great Debate")

    # Start the debate
    user_input = st.chat_input("Initiate the debate with a topic:")
    if st.button("Start Debate"):
        if not user_input:
            user_input = """Who has contributed more to the greater
            advancement of Technology in society? Each bot will try to prove its superiority"""

        simulate_debate(user_input)

    # Display judge's verdict
    if st.button("Judge the Debate"):
        winner, summary = judge()
        st.subheader("Judge's Verdict")
        st.write(winner)
        st.write(summary)


if __name__ == "__main__":
    main()
