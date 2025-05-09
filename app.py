import streamlit as st
import asyncio
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
from main import bot, Deps, messages


st.title("Basic Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state

def display_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

async def get_bot_response(user_input: str) -> str:

    response = await bot.run(user_prompt=user_input, message_history=messages, deps=Deps)
    result = response.output
    return result


display_messages()


user_input = st.chat_input("You: ")

if user_input:
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    bot_response = asyncio.run(get_bot_response(user_input))

    st.session_state.messages.append({"role": "assistant", "content": bot_response})

    with st.chat_message("assistant"):
        st.write(bot_response)

    messages.append(ModelRequest(parts=[UserPromptPart(content=user_input)]))
    messages.append(ModelResponse(parts=[TextPart(content=bot_response)]))