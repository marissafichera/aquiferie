import os
import openai

OPENAI_API = os.getenv('OPENAI_API')
print(f'my OPENAI API KEY ={OPENAI_API}')
client = openai.OpenAI(api_key=OPENAI_API)