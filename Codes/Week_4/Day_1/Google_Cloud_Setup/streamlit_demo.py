import streamlit as st

# Title of the app
st.title("Welcome to My Streamlit App 🚀")

# Subtitle
st.subheader("A Simple Web App to Greet Users")

# User Input
name = st.text_input("Enter your name:", "")

# Button to trigger greeting
if st.button("Greet Me"):
    if name:
        st.success(f"Hello, {name}! 🎉")
    else:
        st.warning("Please enter your name to get a greeting.")

# Display an image
st.image("https://upload.wikimedia.org/wikipedia/en/8/80/Wikipedia-logo-v2.svg", caption="Wikipedia Image")

# Additional Markdown Text
st.markdown("""
### Features:
- 📝 Enter your name
- 🎉 Get a personalized greeting
- 🖼️ Display an image from internet
""")
