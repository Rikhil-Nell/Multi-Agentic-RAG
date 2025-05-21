from pydantic_ai import Agent, RunContext
# from pydantic_ai.models.groq import GroqModel, GroqModelName, GroqModelSettings
# from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.models.openai import OpenAIModel, OpenAIModelName, OpenAIModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import ModelMessage
from pydantic import BaseModel, Field
from supabase import create_client, Client
from openai import AsyncOpenAI
import streamlit as st
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from dataclasses import dataclass
from typing import List, Any

from settings import Settings
from dictionary import dictionary_api

settings = Settings()

console = Console()

# model_name : GroqModelName = "llama-3.3-70b-versatile"

# model = GroqModel(
#     model_name=model_name,
#     provider=GroqProvider(api_key=st.secrets["GROQ_API_KEY"])
# )

# model_settings = GroqModelSettings(
#     temperature=0.8,
#     top_p=1
# )

model_name : OpenAIModelName = "gpt-3.5-turbo"

model = OpenAIModel(
    model_name=model_name,
    provider=OpenAIProvider(api_key=st.secrets["OPENAI_API_KEY"])
)

model_settings = OpenAIModelSettings(
    temperature=0.8,
    top_p=1
)

@dataclass
class Deps:
    supabase_client: Client
    openai_client: AsyncOpenAI

with open("prompt.txt","r") as f:
    prompt = f.read()

bot = Agent(
    model=model,
    model_settings=model_settings,
    system_prompt=prompt,
    deps_type=Deps,
    retries=2,
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536
    
@bot.tool(retries=3)
async def retrieve_relevant_documentation(ctx: RunContext[Deps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        console.rule("[bold blue]🔍 Retrieving Documentation")
        console.log(f"[cyan]User query:[/] {user_query}")

        st.toast("🔍 Starting document retrieval...")

        # Get the embedding
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        console.log("[green]✅ Embedding acquired.")
        st.toast("✅ Got embedding for query")

        # Query Supabase
        result = ctx.deps.supabase_client.rpc(
            'match_documents',
            {
                'query_embedding': query_embedding,
                'match_count': 5
            }
        ).execute()

        if not result.data:
            console.log("[yellow]⚠️ No relevant documents found.")
            st.toast("⚠️ No relevant documentation found.")
            return "No relevant documentation found."

        console.log(f"[green]✅ Retrieved {len(result.data)} documents.")
        st.toast("📄 Formatting top 5 matches...")

        formatted_chunks = []
        for idx, doc in enumerate(result.data, start=1):
            chunk_text = f"""
            # Title: {doc['title']}

            Content: {doc['content']}

            Summary: {doc['summary']}
            """
            formatted_chunks.append(chunk_text)
            console.log(f"[green]✔️ Chunk {idx} formatted.")

        console.rule("[bold green]📚 Retrieval Complete")
        st.toast("✅ Documentation ready.")

        return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:
        console.log(Panel(f"[red bold]❌ Error retrieving documentation:\n{str(e)}", title="Error"))
        st.toast("❌ Error during documentation retrieval.")
        return f"Error retrieving documentation: {str(e)}"

@bot.tool(retries=3)
async def call_dictionary(ctx : RunContext[Deps], word : str) -> str:
    """Get the Merriam Webster dictionary meaning of word. Only to be used when especially asked by the user

    Args:
        ctx: The context.
        word: word to be searched
    """

    try:
        console.rule(f"[bold blue]📚 Dictionary Lookup: {word}")
        st.toast(f"🔍 Looking up: {word}")

        meaning: str = await dictionary_api(search_word=word)

        console.log("[green]✅ Retrieved dictionary meaning.")
        st.toast("✅ Got dictionary definition.")
        return meaning

    except Exception as e:
        console.log(Panel(f"[red bold]❌ Dictionary lookup failed:\n{str(e)}", title="Error"))
        st.toast("❌ Failed to fetch dictionary meaning.")
        return f"Error retrieving dictionary definition: {str(e)}"


messages : List[ModelMessage] = []

# Terminal Test code
if __name__ == "__main__":

    from TestHarness import TerminalChatTest
    import asyncio
    
    openai_client = AsyncOpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    supabase: Client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

    test = TerminalChatTest(agent=bot, deps=Deps(supabase_client=supabase, openai_client=openai_client))
    asyncio.run(test.chat())
