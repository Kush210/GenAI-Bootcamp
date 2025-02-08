import streamlit as st
from openai import OpenAI
from PIL import Image

from diffusers import StableDiffusionImg2ImgPipeline
import torch

import requests
from dotenv import load_dotenv
from io import BytesIO
import tempfile
import os
from diffusers import StableDiffusionImg2ImgPipeline

device = "cuda"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id_or_path, torch_dtype=torch.float16
)
pipe = pipe.to(device)
# Load environment variables
load_dotenv()

# Initialize OpenAI client
# Ensure you have an API key set in your `.env` file or environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Check if the context is already available in session state
if "audio_file" not in st.session_state:
    st.session_state["audio_file"] = None
if "image" not in st.session_state:
    st.session_state["image"] = None
if "mod_audio_file" not in st.session_state:
    st.session_state["mod_audio_file"] = None
if "mod_image" not in st.session_state:
    st.session_state["mod_image"] = None


# Function to transcribe audio using OpenAI Whisper API
def transcribe_audio(audio_file_path):
    """
    This function should:
    - Take the path of the saved audio file as input.
    - Use OpenAI Whisper API to transcribe the audio into text.
    - Return the transcribed text or raise an error if transcription fails.

    You need to:
    - Handle audio file reading logic.
    - Use the OpenAI API `client.audio.transcriptions.create` method.
    - Make sure the model parameter is set (e.g., "whisper-1").
    """
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )
            print(transcript.text)
            return transcript.text
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None


# Function to generate an image using DALL-E 3
def generate_image(prompt, modify=False, img_path=None):
    """
    This function should:
    - Take a textual prompt as input.
    - Use OpenAI DALL-E API to generate an image based on the prompt.
    - Return the image URL.

    You need to:
    - Use the OpenAI API `client.images.generate` method.
    - Ensure parameters like model, prompt, size, and n are set correctly.
    - Handle API response and errors gracefully.
    """
    if not modify:
        try:
            response = client.images.generate(
                model="dall-e-3", prompt=prompt, size="1024x1024", n=1
            )
            return response.data[0].url
        except Exception as e:
            print(f"Error during image generation: {e}")
            return None
    elif img_path is not None:

        response = requests.get(img_path)
        init_image = Image.open(BytesIO(response.content))
        images = pipe(
            prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5
        ).images
        return images[0]

        # if modify and img_path is not None:
        #     print("using dalle-2")
        #     response = client.images.edit(
        #         model="dall-e-2",
        #         image=open(img_path, "rb"),
        #         mask=open("mask.png", "rb"),
        #         prompt="A sunlit indoor lounge area with a pool containing a flamingo",
        #         n=1,
        #         size="1024x1024",
        #     )


# Main Streamlit app
def main():
    """
    Main app logic:
    - Guide the user to upload or provide an audio input.
    - Call the transcription function (`transcribe_audio`) to convert audio to text.
    - Use the transcription to generate an image via `generate_image`.
    - Display the generated image in the Streamlit app.

    You need to:
    - Implement audio file handling (uploading/saving).
    - Handle the logic to call transcription and image generation functions.
    - Display error messages in case of failures.
    """
    st.title("Speech-to-Image Generator")

    # Audio input/upload
    st.write("Provide your audio input describing the image:")
    # User decides how to handle file upload, saving, and reading.

    # file uploader or audio recording widget.
    audio_file = st.audio_input("Record a voice message")

    if audio_file is not None:
        # User decides file-saving logic (e.g., saving as "temp_audio.mp3").

        # save the audio file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
            tmpfile.write(audio_file.read())
            if st.session_state["audio_file"] is None:
                st.session_state["audio_file"] = tmpfile.name

        temp_audio_path = st.session_state["audio_file"]
        # temp_audio_path = "Recording.mp3"  # Replace with actual saving logic.

        # Call the transcription function

        transcription = transcribe_audio(temp_audio_path)
        if transcription:
            st.success(f"Transcription: {transcription}")

            # Call the image generation function
            if st.session_state["image"] is None:
                image_url = generate_image(transcription)
                # Fetch and display the generated image
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content))

                # Save the image to a temporary file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".png"
                ) as tmpfile:
                    # Save the image to the temporary file
                    image.save(tmpfile, format="PNG")
                    if st.session_state["image"] is None:
                        st.session_state["image"] = tmpfile.name
                temp_image_path = st.session_state["image"]

            st.image(
                st.session_state["image"],
                caption="Generated Image",
                use_container_width=True,
            )

            # wait till the user uploads an audio file for modification
            st.write("Modify the generated image by providing additional audio input:")

            # Allow user to modify the image
            mod_audio_file = st.audio_input(
                "Upload an audio file for modifying the generated image",
            )

            if mod_audio_file:

                # save the audio file to a temporary location
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp3"
                ) as tmpfile:
                    tmpfile.write(mod_audio_file.read())
                    temp_audio_path = tmpfile.name

                mod_transcription = transcribe_audio(temp_audio_path)

                if mod_transcription:
                    st.success(f"Transcription: {mod_transcription}")

                    # Call the image generation function
                    mod_image_url = generate_image(
                        mod_transcription,
                        modify=True,
                        img_path=st.session_state["image"],
                    )

                    if mod_image_url:
                        # Fetch and display the generated image
                        response = requests.get(mod_image_url)
                        mod_image = Image.open(BytesIO(response.content))

                        # Display the original and modified images side by side
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(
                                st.session_state["image"],
                                caption="Original Image",
                                use_container_width=True,
                            )
                        with col2:
                            st.image(
                                mod_image,
                                caption="Modified Image",
                                use_container_width=True,
                            )
                    else:
                        st.error("Image generation failed.")
                else:
                    st.error("Transcription failed or returned no valid text.")
        else:
            st.error("Transcription failed or returned no valid text.")
    else:
        st.info("Upload an audio file to get started.")


if __name__ == "__main__":
    main()
