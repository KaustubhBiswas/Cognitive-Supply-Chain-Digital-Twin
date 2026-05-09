import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq

load_dotenv()  # loads GROQ_API_KEY from .env

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

model_name = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
model = ChatGroq(model=model_name, temperature=0)

agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

print(agent.invoke({"messages": [{"role": "user", "content": "What is the weather in San Francisco?"}]}))