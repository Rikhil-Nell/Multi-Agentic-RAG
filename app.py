import asyncio

import streamlit as st
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
from openai import AsyncOpenAI
from supabase import create_client, Client

from main import bot, Deps, messages
from docs import process_and_store_document
from settings import Settings

# --- SETTINGS & CLIENT INITIALIZATION ---
settings = Settings()

openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
supabase: Client = create_client(settings.supabase_url, settings.supabase_key)

deps=Deps(supabase_client=supabase, openai_client=openai_client)

st.set_page_config(page_title="Multi-Agentic RAG", layout="wide")

# ---- Sidebar Navigation ----
st.sidebar.header("📚 Navigation")
selected_page = st.sidebar.radio("Go to", ["About & Uploads", "Chat with Bot"])

# ---- Session State ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# ---- ABOUT & UPLOAD PAGE ----
if selected_page == "About & Uploads":
    st.title("🤖 Multi-Agentic RAG System")
    st.markdown("""
    Welcome to the prototype of a **multi-agent Retrieval-Augmented Generation** chatbot system.

    **What you can do here:**
    - Upload up to **3 PDF files**
    - Files are processed in-memory and not stored
    - Switch to the **Chat with Bot** tab to begin chatting

    ---
    """)

    # PDF Upload section
    uploaded_files = st.file_uploader("📤 Upload PDF files", type="pdf", accept_multiple_files=True)

    # Validation logic
    valid_files = []
    if uploaded_files:
        if len(uploaded_files) > 3:
            st.error("⚠️ Please upload at most 3 PDFs.")
        else:
            for file in uploaded_files:
                valid_files.append(file)
                st.success(f"✅ '{file.name}' accepted.")

        # Store validated files in session state
        st.session_state.uploaded_files = valid_files

        for file in valid_files:
            asyncio.run(process_and_store_document(file=file))

    # Footer links
    st.sidebar.markdown("---")
    st.sidebar.markdown("Made with ❤️ by [Rikhil Nellimarla](https://github.com/Rikhil-Nell) • [LinkedIn](https://www.linkedin.com/in/rikhil-nellimarla?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BrX3QCeOFSFSz%2BwZHwitRuQ%3D%3D)")

# ---- CHAT PAGE ----
elif selected_page == "Chat with Bot":
    st.title("💬 Chat with the Multi-Agentic Bot")
    st.write("Start a conversation powered by your uploaded PDFs.")

    def display_messages():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    async def get_bot_response(user_input: str) -> str:
        response = await bot.run(user_prompt=user_input, deps=deps, message_history=messages)
        return response.output

    display_messages()

    user_input = st.chat_input("Ask me anything...")
    if user_input:
        # User message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Bot response
        bot_response = asyncio.run(get_bot_response(user_input))
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant"):
            st.markdown(bot_response)

        # Message tracking
        messages.append(ModelRequest(parts=[UserPromptPart(content=user_input)]))
        messages.append(ModelResponse(parts=[TextPart(content=bot_response)]))
