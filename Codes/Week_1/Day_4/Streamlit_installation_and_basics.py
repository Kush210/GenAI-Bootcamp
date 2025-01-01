import streamlit as st 
import pandas as pd

st.write("Hey there") 

st.title("First app")

st.header("THis is my first app")

st.subheader("Description goes here")

code = \
'''
#include<bits/stdc++.h>
using namespace std; 
int main(){

}
'''

st.code(code, language="cpp")

st.checkbox("Agree to the terms and condtions")
op = st.button("Press me")

options =  ["Red","Blue","Green","Yellow","Orange"]

st.radio("Choose any one ",options)

st.selectbox("THis is a select box ",options=options)
st.multiselect("Choose any ",options)

inp = st.color_picker("choose any color")
