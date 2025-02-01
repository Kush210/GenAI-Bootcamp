import streamlit as st
import pandas as pd

st.write("Hey there")

st.title("First app")

st.header("This is my first app")

st.subheader("Description goes here")

code = """
#include<bits/stdc++.h>
using namespace std; 
int main(){

}
"""

st.code(code, language="cpp")

st.checkbox("Agree to the terms and condtions")
op = st.button("Press me")

if op:
    st.write("Hello there!")

options = ["Red", "Blue", "Green", "Yellow", "Orange"]

st.radio("Choose any one ", options)

st.selectbox("This is a select box ", options=options)
st.multiselect("Choose any ", options)

inp = st.color_picker("choose any color")
input_text = st.text_input("What is your name?")
if input_text:
    st.write(f"Hello {input_text}")
    print(f"Bye {input_text}")
