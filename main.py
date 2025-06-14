import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv('OPENROUTER_API_KEY')

client = OpenAI(
    base_url='https://openrouter.ai/api/v1',
    api_key=api_key,
)

stream = client.chat.completions.create(
    extra_headers={
        # 'HTTP-Referer': '<YOUR_SITE_URL>',  # Optional. Site URL for rankings on openrouter.ai.
        # 'X-Title': '<YOUR_SITE_NAME>',  # Optional. Site title for rankings on openrouter.ai.
    },
    # model='openai/gpt-4o',
    model='deepseek/deepseek-chat-v3-0324:free',
    messages=[{'role': 'user', 'content': 'What is the meaning of life?'}],
    stream=True,
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='', flush=True)
print()
