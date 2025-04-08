import os
import openai

OPENAI_API = os.getenv('OPENAI_API')
client = openai.OpenAI(api_key=OPENAI_API)