import gradio as gr 
import requests as re 

def query(user_input):
    url = 'http://localhost:11434/api/chat'
    payload = {
        "model" : "gemma:2b",
        "messages" : [{
            "role" : "user",
            "content" : user_input
        }],
        "stream" : False
    }

    response = re.post(url = url, json = payload)

    return response.json()['message']['content']


def chatbot_interface(user_input, chat_history):

    chat_history.append(("User",user_input))

    try:
        res = query(user_input)
    except Exception as e:
        res = f"Error : {e}"

    chat_history.append(("Assistant", res))

    return chat_history,chat_history,""


with gr.Blocks() as chatbot_ui:

    chatbot = gr.Chatbot(label = "Chat", height = 400)
    user_input = gr.Textbox(label = "Input", placeholder = "Your message here", lines = 1)
    send_button = gr.Button("Send")
    reset_button = gr.Button("Reset")

    chat_history = gr.State([])
    send_button.click(
        chatbot_interface,
        inputs = [user_input,chat_history],
        outputs = [chatbot, chat_history, user_input]

    )

    user_input.submit(
        chatbot_interface,
        inputs = [user_input,chat_history],
        outputs = [chatbot, chat_history, user_input]
    )

    reset_button.click(lambda : ([], []), inputs = None, outputs = [chatbot, chat_history])

chatbot_ui.launch()