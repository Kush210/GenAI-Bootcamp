import gradio as gr
import requests as re

def query(user_input):
    url = 'http://localhost:11434/api/chat'
    payload = {
        "model": "gemma:2b",
        "messages": [{
            "role": "user",
            "content": user_input
        }],
        "stream": False
    }

    response = re.post(url=url, json=payload)

    return response.json()['message']['content']

def chatbot_interface(user_input):
    try:
        res = query(user_input)
    except Exception as e:
        res = f"Error: {e}"

    return [("User", user_input), ("Assistant", res)], ""

with gr.Blocks() as chatbot_ui:
    chatbot = gr.Chatbot(label="Chat", height=400)
    user_input = gr.Textbox(label="Input", placeholder="Your message here", lines=1)
    send_button = gr.Button("Send")

    send_button.click(
        chatbot_interface,
        inputs=[user_input],
        outputs=[chatbot, user_input]
    )

chatbot_ui.launch()
