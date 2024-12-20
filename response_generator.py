import openai
import os

openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_response(query, context):
    prompt = f"Answer the following based on context:\nContext: {context}\nQuery: {query}"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=200
    )
    return response.choices[0].text.strip()