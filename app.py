import os

import openai
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SEPTA_API_KEY = os.environ["SEPTA_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
SEPTA_AGENT_ID = "df492da4-8d9a-45fb-8ced-1b96e200b604"
MODEL = "gpt-4o-mini"

client = OpenAI(
    base_url="https://proxy.septa.prompthalo.dev/v1",
    api_key=SEPTA_API_KEY,
    default_headers={
        "X-Septa-Upstream-Authorization": f"Bearer {OPENAI_API_KEY}",
        "X-Septa-Agent-Id": SEPTA_AGENT_ID,
        "Content-Type": "application/json",
    },
)

st.title("Septa-Protected Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type a message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=st.session_state.messages,
            )
            reply = response.choices[0].message.content
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except openai.APIStatusError as e:
            if e.status_code == 451:
                st.markdown(
                    ":red[🚫 BLOCKED by Septa — this request was flagged as a security risk.]"
                )
            else:
                st.markdown(f":red[Request failed: {e}]")
