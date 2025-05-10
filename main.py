from pydantic_ai import Agent, RunContext
from pydantic_ai.models.groq import GroqModel, GroqModelName, GroqModelSettings
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.messages import ModelMessage
from pydantic import BaseModel, Field

from dataclasses import dataclass
from settings import Settings
from typing import List

from dictionary import dictionary_api

settings = Settings()

model_name : GroqModelName = "llama-3.3-70b-versatile"

model = GroqModel(
    model_name=model_name,
    provider=GroqProvider(api_key=settings.groq_api_key)
)

model_settings = GroqModelSettings(
    temperature=0.8,
    top_p=1
)

@dataclass
class Deps:
    pass

with open("prompt.txt","r") as f:
    prompt = f.read()

bot = Agent(
    model=model,
    model_settings=model_settings,
    system_prompt=prompt,
    deps=Deps(),
    retries=2,
)


@bot.tool
async def call_dictionary(ctx : RunContext[Deps], word : str) -> str:
    """Get the Merriam Webster dictionary meaning of word. Only to be used when especially asked by the user

    Args:
        ctx: The context.
        word: word to be searched
    """

    meaning : str = await dictionary_api(search_word=word)
    return meaning


messages : List[ModelMessage] = []


# Terminal Test code
if __name__ == "__main__":

    from TestHarness import TerminalChatTest
    import asyncio

    test = TerminalChatTest(agent=bot, deps=Deps)

    asyncio.run(test.chat())