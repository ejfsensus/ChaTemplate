import os
import requests
import chainlit as cl
from openai import AsyncOpenAI
import spacy

# Load the small English NLP model
nlp = spacy.load("en_core_web_sm")

# Load environment variables
client = AsyncOpenAI(base_url="https://api.aimlapi.com/v1", api_key=os.getenv('OPENAI_API_KEY'))
cl.instrument_openai()

# Define model settings
settings = {
    "model": "MiniMax-Text-01",
    "temperature": 0.1,
}

# Brave Summarizer Search function
def brave_summarizer_search(query: str, api_key: str, num_results: int = 5):
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key,
    }
    params = {
        "q": query,
        "count": num_results,
    }
    response = requests.get("https://api.search.brave.com/res/v1/summarizer", headers=headers, params=params)
    response.raise_for_status()
    return response.json()

# Function to detect if a query should trigger a web search
def should_use_web_search(query: str) -> bool:
    doc = nlp(query)
    for token in doc:
        if token.lemma_ in ["who", "what", "where", "when", "why", "how", "find", "show", "look"]:
            return True
    if any(ent.label_ == "DATE" for ent in doc.ents):
        return True
    return False

# on_message function to handle incoming messages
@cl.on_message
async def on_message(message: cl.Message):
    user_query = message.content
    
    # Use NLP to determine if a web search is necessary
    if should_use_web_search(user_query):
        search_results = brave_summarizer_search(user_query, api_key=os.getenv('BRAVE_API_KEY'))
        if search_results.get('summary'):
            search_context = f"Search Summary: {search_results['summary']}\n"
        else:
            search_context = "No relevant search results found.\n"
    else:
        search_context = ""
    
    # Use the LLM with search data as context
    response = await client.chat.completions.create(
        messages=[
            {"content": "You are a helpful bot, you always reply in Spanish", "role": "system"},
            {"content": f"{search_context}{user_query}", "role": "user"}
        ],
        **settings
    )
    response_content = response.choices[0].message.content

    await cl.Message(content=response_content).send()