# In src/travel_agent/main.py
import os
import sys
from dotenv import load_dotenv
from getpass import getpass
import google.generativeai as genai

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Load environment variables
load_dotenv()

def configure_api_keys():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    serpapi_api_key = os.getenv("SERPAPI_API_KEY")

    if not google_api_key:
        os.environ["GOOGLE_API_KEY"] = getpass("Enter your Google API Key: ")
    if not tavily_api_key:
        os.environ["TAVILY_API_KEY"] = getpass("Enter your Tavily API Key: ")
    if not serpapi_api_key:
        os.environ["SERPAPI_API_KEY"] = getpass("Enter your SerpApi API Key: ")
    
    # Configure the genai library once at the start.
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

configure_api_keys()

# Now, import your graph after the environment is configured.
from src.travel_agent.graph import build_travel_graph

def run_chatbot():
    travel_planner = build_travel_graph()
    # ... rest of your chatbot code
    
if __name__ == '__main__':
    run_chatbot()