import gradio as gr
from openai import OpenAI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
<<<<<<< HEAD
import pandas as pd
=======
>>>>>>> vettura/main
import os

# Load environment variables
load_dotenv()

# Initialize OpenAI client
<<<<<<< HEAD
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define models - User should add their models and identifiers here
models = {
    "GPT-4": "gpt-4",
    "GPT-4o": "gpt-4o",
    "GPT-4o-mini": "gpt-4o-mini",
    "GPT3.5 Turbo": "gpt-3.5-turbo-0125",
    # Example: "Model Name": "model-identifier",
}


# Define schemas for parsing responses
class ModelSchema(BaseModel):
    model_name: str
    model_response: str
    model_vote: str

=======
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Define models - User should add their models and identifiers here
models = {
    # Example: "Model Name": "model-identifier",
}

# Define schemas for parsing responses
class ModelSchema(BaseModel):
    model_name: str
    model_rating: int
>>>>>>> vettura/main

class JudgeSchema(BaseModel):
    models_rating: List[ModelSchema]
    winner: str

<<<<<<< HEAD

Dashboard = JudgeSchema(models_rating=[], winner="")


# Function to query a model
def query(user_input, model_id):
=======
# Function to query a model
def query(user_input, model_id, judge=None):
>>>>>>> vettura/main
    """
    Query a specific model with the given input.
    If judge is provided, include judging logic.
    """
<<<<<<< HEAD
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": user_input}],
            temperature=0.7,
            max_tokens=250,
        )
    except Exception as e:
        return f"Error with {model_id}: {e}"
    return response.choices[0].message.content


# Function to judge responses
def chat_interface(history, user_message):
    """
    Evaluate the responses from different models and determine the best answer.
    """

=======
    pass  # Add query logic here

# Function to judge responses
def judge(model_name, history, user_message):
    """
    Evaluate the responses from different models and determine the best answer.
    """
>>>>>>> vettura/main
    # Function to get responses from all models
    def chat(history, user_message):
        """
        Generate responses from all models and append them to the chat history.
        """
<<<<<<< HEAD
        model_response = {}
        for model_name, model_id in models.items():
            response = query(user_message, model_id)
            history.append((model_name, response))
            model_response[model_id] = response
        return model_response, history

    history.append(("User", user_message))
    print("Generating model responses...")
    # Generate responses and evaluate
    model_response, history = chat(history, user_message)

    votes = {}
    # Judge to vote the best response among the other models
    for model_name in models:
        print(model_name + " is judging...")
        # Construct prompt for judging
        prompt = f"""Give a single word answer.
        You are a judge tasked with evaluating the performance of AI models.
        Vote the best model based on their responses to the user query. There cannot be a tie.
        Only, provide the name of the winning model.\n"""
        prompt += f" User Query: {user_message}\n"

        for model_id, response in model_response.items():
            if model_id == models[model_name]:
                judge_model = ModelSchema(
                    model_name=model_name, model_response=response, model_vote=""
                )
                continue
            else:
                prompt += f"{model_id}: {response}\n"

        # Query the judge model
        response = query(prompt, models[model_name])
        print(model_name + " votes " + response)
        history.append((model_name, " votes " + response))
        judge_model.model_vote = response
        votes[response] = votes.get(response, 0) + 1
        Dashboard.models_rating.append(judge_model)

    # Format and append judging results
    winner = max(votes, key=votes.get)
    Dashboard.winner = winner
    print("Winner is " + winner)
    # Pretty print the Dashboard schema as a table
    df = pd.DataFrame([model.model_dump() for model in Dashboard.models_rating])

    history.append(("System", "Winner is " + winner))
    return history, winner, df

=======
        pass  # Add chat logic here

    # Generate responses and evaluate
    model_response, history = chat(history, user_message)

    # Construct prompt for judging - User should define their own judging prompt here
    prompt = """
    # Example judging prompt:
    You are a judge tasked with evaluating the performance of AI models.
    Rate the models based on their responses to the user query on a scale of 10.
    Also, provide the name of the winning model.
    User Query: {user_message}
    """
    for model_id, response in model_response.items():
        if model_id == models[model_name]:
            continue
        else:
            prompt += f"{model_id}: {response}\n"

    # Query the judge model
    final_response = query(prompt, models[model_name], judge=model_name)

    # Format and append judging results
    formatted_response = f"## {model_name} (Judge)\n"
    history.append((None, formatted_response))
    return history, ""
>>>>>>> vettura/main

# Main function - User should write the Gradio interface here
def main():
    """
    Build the Gradio interface for interacting with multiple AI models.
    Add your own logic for interface elements.
    """
<<<<<<< HEAD
    with gr.Blocks() as chatbot_ui:
        chatbot = gr.Chatbot(label="Chat", height=400)
        user_input = gr.Textbox(label="Input", placeholder="Your message here", lines=1)
        send_button = gr.Button("Send")
        dashboard = gr.Dataframe(
            label="Dashboard", headers=["Model Name", "Response", "Vote"]
        )
        winner = gr.Textbox(label="Winner", placeholder="Winner")

        send_button.click(
            chat_interface,
            inputs=[chatbot, user_input],
            outputs=[chatbot, winner, dashboard],
        )

    chatbot_ui.launch(share=True)

    prompts = {
        "Who was the president of the India in 2002",
        "Which country has the largest coal production in the world?",
        "What are the top 3 countries with the highest GDP in 2012?",
        "What are the top 3 countries with lowest population density in 2012?",
        "Which country are still part of the British Commonwealth?",
    }

=======
    pass  # Add Gradio interface code here
>>>>>>> vettura/main

# Entry point of the application
if __name__ == "__main__":
    main()
