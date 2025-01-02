import streamlit as st 

if "c" not in st.session_state:
    st.session_state.c = 0


button = st.button("Increase me")

if button:
    st.session_state.c +=1 
    st.write(f"Counter: {st.session_state.c}")