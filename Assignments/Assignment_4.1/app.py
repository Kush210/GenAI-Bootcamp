from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
import streamlit as st
import traceback
from langchain_community.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


def setup_agent():

    serper_search = GoogleSerperAPIWrapper(
        serper_api_key=os.getenv("SERPER_DEV_API_KEY")
    )

    # define serper search tool
    serper_search_tool = Tool(
        name="Serper Search",
        func=serper_search.run,
        description="A tool for performing web searches using Serper.dev.",
    )

    llm_openai = ChatOpenAI(
        model_name="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Load standard tools
    standard_tools = load_tools(
        ["wikipedia"], llm=llm_openai, allow_dangerous_tools=True
    )

    tools = standard_tools + [serper_search_tool]

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools,
        llm_openai,
        agent="zero-shot-react-description",
        handle_parsing_errors=True,
        max_iterations=15,  # Set to an appropriate number
        memory=memory,
        verbose=True,
    )
    return agent


# Initialize Session State
def initialize_session_state():
    """Initializes required variables in the session state."""
    if "agent" not in st.session_state:
        st.session_state["agent"] = None


def main():

    initialize_session_state()

    st.session_state["agent"] = setup_agent()

    st.title("What's Trending")

    user_input = st.chat_input("Enter a personality you would like to know about..")

    if user_input:
        if not st.session_state.get("agent"):
            st.error("Error establishing agent.")
        else:
            agent = st.session_state["agent"]
            with st.spinner("Fetching latest news..."):
                # Run the agent with the properly formatted input
                agent_query = f"""You are a news summarization agent. You are provided with a variable called {user_input} that contains the name of a public figure
                                or personality as specified by the user. Your task is to retrieve and synthesize the most important recent news about this personality and
                                present it as a clear, well-structured article in markdown format in three paragraphs. """
                response = agent.invoke(agent_query)
                reply = response["chat_history"][-1].content
                print("-----" * 20, reply)
                if not reply.startswith("#"):
                    if "markdown" in reply:
                        reply = reply.split("markdown", 1)[1]
                        print("-----" * 20, reply)
                st.markdown(reply)


if __name__ == "__main__":
    main()
