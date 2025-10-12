import os

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from prompts.model_prompts import SYSTEM_PROMPT


def setup_agent() -> Agent:
    """Sets up the model to be used and returns a Pydantic Agent

    Returns:
        Agent: Pydantic Agent
    """

    load_dotenv()

    model = OpenAIChatModel(
        model_name="sonar",
        provider=OpenAIProvider(base_url="https://api.perplexity.ai", api_key=os.getenv("PPLX_API_KEY", "")),
        settings={"temperature": 0.1},
    )

    agent = Agent(model, system_prompt=SYSTEM_PROMPT)
    return agent


if __name__ == "__main__":
    agent = setup_agent()
    response = agent.run_sync("Hi")
    print(response.output)
