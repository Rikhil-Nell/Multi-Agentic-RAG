from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, UserPromptPart, TextPart
from typing import List

class TerminalChatTest:
    def __init__(self, agent, deps):
        self.agent = agent
        self.deps = deps
        self.messages : List[ModelMessage] = []

    async def chat(self):
        while True:
            user_input = input("You: ")
            response = await self.agent.run(user_prompt=user_input, message_history=self.messages, deps=self.deps)
            self.messages.append(ModelRequest(parts=[UserPromptPart(content=user_input)]))
            self.messages.append(ModelResponse(parts=[TextPart(content=response.output)]))
            print("Bot:", response.output)
            if user_input == "exit":
                break
