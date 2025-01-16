import streamlit as st 

non_state_var = 0
if "state_var" not in st.session_state:
    st.session_state.state_var = 0


button = st.button("Increase Variables")

if button:
    non_state_var +=1
    st.session_state.state_var +=1 
    st.write(f"Non State Variable: {non_state_var}")
    st.write(f"State Variable: {st.session_state.state_var}")