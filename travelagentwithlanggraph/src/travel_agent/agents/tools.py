import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain.tools import Tool
from getpass import getpass
import json
import serpapi
from langchain_tavily import TavilySearch

# TavilySearch Tool
tavily_tool = TavilySearch(max_results=2)

# Your existing code continues below for the tool definitions
# ...
# Flight Search Tool
def search_flights(departure_airport: str, arrival_airport: str, outbound_date: str, return_date: str = None, adults: int = 1, children: int = 0) -> str:
    """
    Search for flights using Google Flights engine.
    ...
    """
    params = {
        'api_key': os.environ.get('SERPAPI_API_KEY'),
        'engine': 'google_flights',
        'hl': 'en',
        'gl': 'us',
        'departure_id': departure_airport,
        'arrival_id': arrival_airport,
        'outbound_date': outbound_date,
        'return_date': return_date,
        'currency': 'USD',
        'adults': adults,
        'children': children,
        'stops': '1'
    }
    try:
        search = serpapi.search(params)
        results = search.data.get('best_flights', [])
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Flight search failed: {str(e)}"
search_flights_tool = Tool.from_function(
    func=search_flights,
    name="search_flights",
    description="Search for flights using Google Flights engine."
)

# Hotel Search Tool
def search_hotels(location: str, check_in_date: str, check_out_date: str, adults: int = 1, children: int = 0, rooms: int = 1, hotel_class: str = None, sort_by: int = 8) -> str:
    """
    Search for hotels using Google Hotels engine.
    ...
    """
    adults = int(float(adults)) if adults else 1
    children = int(float(children)) if children else 0
    rooms = int(float(rooms)) if rooms else 1
    sort_by = int(float(sort_by)) if sort_by else 8
    params = {
        'api_key': os.environ.get('SERPAPI_API_KEY'),
        'engine': 'google_hotels',
        'hl': 'en',
        'gl': 'us',
        'q': location,
        'check_in_date': check_in_date,
        'check_out_date': check_out_date,
        'currency': 'USD',
        'adults': adults,
        'children': children,
        'rooms': rooms,
        'sort_by': sort_by
    }
    if hotel_class:
        params['hotel_class'] = hotel_class
    try:
        search = serpapi.search(params)
        properties = search.data.get('properties', [])
        if not properties:
            return f"No hotels found. Available data keys: {list(search.data.keys())}"
        return json.dumps(properties[:5], indent=2)
    except Exception as e:
        return f"Hotel search failed: {str(e)}"
search_hotels_tool = Tool.from_function(
    func=search_hotels,
    name="search_hotels",
    description="Search for hotels using Google Hotels engine."
)