import streamlit as st
import time
import llm_chat
graph = llm_chat.graph



config = {"configurable": {"thread_id": "test123"}}
if "messages" not in st.session_state:
    st.session_state.messages = []

def stream_output(message):
    for i in message:
       time.sleep(0.03)
       yield i 




for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("ask agent a question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("responding..."):
            response = graph.invoke(
                {"messages": [( "user",  prompt)]},
                config=config)
            message = response["messages"][-1].content
        response = st.write_stream(stream_output(message))
        st.session_state.messages.append({"role": "assistant", "content": response})