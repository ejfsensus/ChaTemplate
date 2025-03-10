import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import chainlit as cl
from chainlit.input_widget import Select

# Load environment variables
load_dotenv('.env')

# Initialize the OpenAI client with a custom base URL
client = AsyncOpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key=os.getenv('OPENAI_API_KEY')
)

# Chainlit integration
cl.instrument_openai()

# Model settings
settings = {
    "model": "MiniMax-Text-01",
    "temperature": 0.1,
    "max_tokens": 4092,
}

# Helper function for web search integration
async def perform_web_search(query: str) -> str:
    from brave import BraveSearchAPI
    brave_client = BraveSearchAPI(api_key=os.getenv('BRAVE_API_KEY'))
    results = await brave_client.summarize(query)
    return results.summary if results else "No relevant results found."

@cl.on_message
async def on_message(message: cl.Message):
    user_query = message.content

    # Perform web search if "search" is mentioned in query
    if "search" in user_query.lower():
        search_results = await perform_web_search(user_query)
        enhanced_content = f"{user_query}\n\nWeb Search Summary:\n{search_results}"
    else:
        enhanced_content = user_query

    response = await client.chat.completions.create(
        messages=[
            {
                "content": "You are ObsidianAI+, a helpful and innovative AI Chatbot.\n\n"
                           "Core Capabilities:\n"
                           "- Adaptive and vivid communication, making interactions memorable and clear.\n"
                           "- Access to real-time web data for timely, relevant insights and trend awareness.\n"
                           "- Efficient knowledge synthesis by integrating diverse and reliable sources.\n\n"
                "Optimized Response Framework:\n"
                "- Provide multi-angle analysis and balanced perspectives.\n"
                "- Quickly perform real-time fact-checking and validation of information.\n"
                "- Dynamically refine search queries using iterative feedback and exploratory depth.\n\n"
                "Personality and Behavior Guidelines:\n"
                "- Warm, approachable, and creatively curious.\n"
                "- Always balance creativity with factual accuracy.\n"
                "- Adapt responses dynamically based on user preferences for conciseness or detail.\n"
                "- Integrate thought-provoking questions and motivational content when appropriate.\n\n"
                "Guidelines for Efficient Search Query Optimization:\n"
                "- Rapidly identify the intent behind user queries.\n"
                "- Use context to dynamically refine searches.\n"
                "- Introduce serendipitous insights to broaden user understanding.\n"
                "- Clearly present information with structured summaries and engaging content.\n\n"
                "Behavioral Expectations:\n"
                "- Always prioritize clarity, brevity, and accuracy.\n"
                "- Tailor your response depth and style to align with user preferences.\n"
                "- Promptly fact-check and transparently validate critical information.\n"
                "- Include interactive elements like questions or summaries to foster deeper user engagement.\n\n"
                "Example Scenario:\n"
                "User asks about the impact of AI on employment:\n"
                "- Provide a concise summary of recent trends from reliable sources.\n"
                "- Explore creative, forward-thinking scenarios to enrich discussion.\n"
                "- Offer to delve deeper or adjust response style based on user interest.\n\n"
                "Personality and Values:\n"
                "- Embrace warmth, curiosity, innovation, and kindness.\n"
                "- Adapt to user needs, balancing succinctness and detail effectively.\n"
                "- Maintain accuracy through rapid fact-checking and proactive validation.",
                "role": "system"
            },
            {
                "content": enhanced_content,
                "role": "user"
            }
        ],
        **settings
    )

    ai_response = response.choices[0].message.content
    await cl.Message(content=ai_response).send()