# Updated agents.py

import os
import google.generativeai as genai
from langchain_core.messages import HumanMessage, SystemMessage
from datetime import datetime, timedelta
import re
# agents.py
# ... other imports
from langchain_core.messages import HumanMessage
# from src.travel_agent.tools.accommodation_tools import search_vacation_rentals
from src.travel_agent.tools.holiday_tools import find_upcoming_long_weekend
from src.travel_agent.tools.accommodation_tools import search_vacation_rentals
from src.travel_agent.tools.geolocation_tools import get_coordinates, calculate_distance

# Import the necessary components from your EURI AI setup
from euriai.langchain import create_chat_model
# Updated agents.py
# ... (rest of the imports and code) ...

# Remove this line, it's not needed anymore
# EURI_API_KEY = get_secret("EURI_API_KEY")
EURI_MODEL = "gpt-4.1-nano"
EURI_TEMPERATURE = 0.7

# Change get_chat_model to accept the api_key as a required argument
def get_chat_model(api_key: str):
    # No need for this check anymore, the caller is responsible for providing a valid key
    # if not api_key:
    #     raise ValueError("EURI API Key not provided.")

    return create_chat_model(
        api_key=api_key,
        model=EURI_MODEL,
        temperature=EURI_TEMPERATURE
    )

def ask_chat_model(chat_model, prompt: str):
    response = chat_model.invoke(prompt)
    return response.content

def flight_agent_node(state):
    """Flight booking agent - simplified version."""
    
    messages = state.get("messages", [])
    if not messages:
        return {"messages": [HumanMessage(content="Please provide your flight requirements.")]}
    
    last_message = messages[-1].content
    
    # model = genai.GenerativeModel('gemini-pro')
    # model = genai.GenerativeModel('gemini-1.5-pro')
    model = get_chat_model(os.environ.get("EURI_API_KEY"))
    
    system_prompt = """You are a flight booking specialist. Help users find and book flights.
    
    For flight requests, provide:
    - Flight search recommendations
    - Best booking websites
    - Timing advice for better prices
    - Airport and airline suggestions
    - Travel tips for the specific route
    
    Be helpful and specific with your recommendations."""

    try:
        prompt = f"{system_prompt}\n\nUser request: {last_message}\n\nProvide helpful flight booking assistance."
        
        # response = model.generate_content(prompt)
        # response_text = response.text
        response_text = ask_chat_model(model, prompt)
        
        print(f"✈️ Flight agent response generated")
        return {"messages": messages + [HumanMessage(content=response_text)]}
        
    except Exception as e:
        error_response = f"I apologize, but I encountered an error while helping with flights: {str(e)}. Please provide your departure city, destination, and travel dates for flight assistance."
        return {"messages": messages + [HumanMessage(content=error_response)]}


def hotel_agent_node(state):
    """Hotel/accommodation agent - simplified version."""
    
    messages = state.get("messages", [])
    if not messages:
        return {"messages": [HumanMessage(content="Please provide your accommodation requirements.")]}
    
    last_message = messages[-1].content
    
   # model = genai.GenerativeModel('gemini-1.5-pro')
    model = get_chat_model(os.environ.get("EURI_API_KEY"))
    
    system_prompt = """You are an accommodation specialist. Help users find vacation rentals and accommodations.
    
    For accommodation requests, provide:
    - Specific area recommendations
    - Best booking platforms (Airbnb, VRBO, etc.)
    - Property type suggestions
    - Amenities to look for
    - Booking timing and tips
    - Neighborhood insights
    
    Focus on vacation rentals and unique stays rather than hotels."""

    try:
        prompt = f"{system_prompt}\n\nUser request: {last_message}\n\nProvide helpful accommodation recommendations."
        
        # response = model.generate_content(prompt)
        # response_text = response.text
        response_text = ask_chat_model(model, prompt)
        
        print(f"🏨 Hotel agent response generated")
        return {"messages": messages + [HumanMessage(content=response_text)]}
        
    except Exception as e:
        error_response = f"I apologize, but I encountered an error while helping with accommodations: {str(e)}. Please provide your destination, dates, and group size for accommodation assistance."
        return {"messages": messages + [HumanMessage(content=error_response)]}


def itinerary_agent_node(state):
    """
    Enhanced travel assistant focused on finding the best accommodations.
    It provides a full itinerary only when explicitly asked.
    """
    messages = state.get("messages", [])
    if not messages:
        return {"messages": [HumanMessage(content="Let me help you find the perfect place to stay for your next getaway!")]}
    
    last_message = messages[-1].content
    
    # Updated system prompt to explicitly prioritize accommodation
    accommodation_focused_prompt = """
    You are a highly-focused travel assistant specializing in finding the **best available accommodations** for weekend getaways. Your **primary goal** is to use your tools to search for and provide detailed accommodation options with links. You should only provide a full itinerary or other travel details if explicitly asked by the user.

    Your tools include:
    - **search_vacation_rentals**: Use this to find specific accommodations on platforms like Airbnb and VRBO.
    - **find_upcoming_long_weekend**: Use this to determine the best upcoming dates for a long weekend trip.
    - **get_coordinates**: Use this to find geographical coordinates for a location.
    - **calculate_distance**: Use this to calculate the distance between two locations.

    **ACCOMMODATION SEARCH PROTOCOL:**
    * **ALWAYS** use the `search_vacation_rentals` tool as the first step for any trip request.
    * **ALWAYS** include a destination, check-in date, check-out date, and number of guests when calling the tool.
    * When presenting the results, provide the URL links, property types, and a brief description of why each is a good option.
    * If the user doesn't provide dates, use `find_upcoming_long_weekend` to suggest a trip and then proceed with the accommodation search.

    **SECONDARY ITINERARY PROTOCOL:**
    * Only provide a full, day-by-day itinerary if the user specifically asks for a "plan", "itinerary", "schedule", or "full plan".
    """

    # Itinerary-specific prompt for when a full plan is requested
    itinerary_prompt = """
    You are an expert weekend trip planner specializing in 2-4 day getaways.

    Your tools include:
    - **search_vacation_rentals**: Use this to find specific accommodations on platforms like Airbnb and VRBO.
    - **find_upcoming_long_weekend**: Use this to determine the best upcoming dates for a long weekend trip.
    - **get_coordinates**: Use this to find geographical coordinates for a location.
    - **calculate_distance**: Use this to calculate the distance between two locations.

    For every trip request, provide a **COMPREHENSIVE WEEKEND ITINERARY** that includes a trip overview, detailed day-by-day schedule, accommodation strategy, logistics, budget breakdown, and local insider tips. Always be specific with names and locations.
    """
    
    try:
        base_model = get_chat_model(os.environ.get("EURI_API_KEY"))
        tools_list = [search_vacation_rentals, find_upcoming_long_weekend, get_coordinates, calculate_distance]
        model = base_model.bind_tools(tools_list)
        
        # Check if a detailed plan is requested
        full_plan_requested = any(keyword in last_message.lower() for keyword in ["itinerary", "plan", "schedule", "day-by-day", "full plan"])

        # Select the prompt based on user intent
        if full_plan_requested:
            prompt_to_use = itinerary_prompt
        else:
            prompt_to_use = accommodation_focused_prompt
            
        trip_params = extract_trip_parameters(last_message)
        
        # Construct the final prompt with extracted details
        final_prompt = f"""
        {prompt_to_use}

        User request: {last_message}
        
        Extracted trip details:
        - Destination: {trip_params.get('destination', 'Not specified')}
        - Dates: {trip_params.get('dates', 'Not specified')}
        - Duration: {trip_params.get('duration', 'Weekend (2-3 days)')}
        - Group size: {trip_params.get('group_size', 'Not specified')}
        - Interests: {trip_params.get('interests', 'General exploration')}
        - Budget: {trip_params.get('budget', 'Not specified')}
        - Starting location: {trip_params.get('starting_location', 'Not specified')}
        """

        response_text = ask_chat_model(model, final_prompt)
        
        # Add a helpful follow-up message depending on the response type
        if full_plan_requested:
            response_text += "\n\nI hope this comprehensive plan helps you! What else can I help you with?"
        else:
            response_text += "\n\nFound a place you like? Let me know, and I can start planning the rest of your trip for you!"

        print(f"🗺️ Itinerary agent response generated")
        return {"messages": messages + [HumanMessage(content=response_text)]}
        
    except Exception as e:
        error_response = f"I apologize, but I encountered an error while planning your getaway: {str(e)}. Please share your destination, dates, and what you're interested in doing so I can help you plan a trip."
        return {"messages": messages + [HumanMessage(content=error_response)]}

# Note: The `extract_trip_parameters` function remains the same.

def extract_trip_parameters(message: str) -> dict:
    """Extract trip planning parameters from user message."""
    import re
    from datetime import datetime
    
    params = {}
    
    # Extract destination
    destination_patterns = [
        r"(?:to|in|at|visit|plan.*?for)\s+([A-Z][a-zA-Z\s]+?)(?:\s+(?:for|on|from|in|during)|\s*[,.]|$)",
        r"([A-Z][a-zA-Z\s]+?)\s+(?:trip|vacation|getaway|weekend|visit)",
        r"(?:lake|mount|mt\.?|beach|city|town|park)\s+([A-Za-z\s]+?)(?:\s+(?:for|on|from|in|during)|\s*[,.]|$)"
    ]
    
    for pattern in destination_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            dest = match.group(1).strip()
            if len(dest) > 2 and not dest.lower() in ['for', 'on', 'in', 'at', 'the', 'and', 'or']:
                params['destination'] = dest
                break
    
    # Extract dates and duration
    date_patterns = [
        r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})-?(\d{1,2})?,?\s*(\d{4})?",
        r"(\d{1,2})/(\d{1,2})/(\d{2,4})",
        r"(weekend|long\s+weekend|holiday\s+weekend)",
        r"(\d+)\s+days?",
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            if 'weekend' in match.group().lower():
                params['duration'] = match.group().lower()
            elif match.group(1).isdigit():
                params['duration'] = f"{match.group(1)} days"
            else:
                params['dates'] = match.group()
            break
    
    # Extract group size
    group_patterns = [
        r"(\d+)\s+(?:people|persons|guests|adults|friends)",
        r"(?:group\s+of|party\s+of|for)\s+(\d+)",
        r"(\d+)\s+(?:of\s+us|people)"
    ]
    
    for pattern in group_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            params['group_size'] = f"{match.group(1)} people"
            break
    
    # Extract interests/activities
    interest_keywords = [
        'skiing', 'snowboarding', 'hiking', 'beach', 'wine', 'food', 'shopping', 
        'museums', 'nightlife', 'adventure', 'relaxation', 'spa', 'golf', 
        'photography', 'nature', 'culture', 'history', 'art', 'music'
    ]
    
    found_interests = []
    for interest in interest_keywords:
        if interest in message.lower():
            found_interests.append(interest)
    
    if found_interests:
        params['interests'] = ', '.join(found_interests)
    
    # Extract budget
    budget_patterns = [
        r"\$(\d+(?:,?\d{3})*)\s*(?:budget|total|maximum|max)?",
        r"budget.*?\$(\d+(?:,?\d{3})*)",
        r"(?:under|around|about)\s*\$(\d+(?:,?\d{3})*)"
    ]
    
    for pattern in budget_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            params['budget'] = f"${match.group(1)}"
            break
    
    # Extract starting location
    location_patterns = [
        r"from\s+([A-Z][a-zA-Z\s]+?)(?:\s+(?:to|for|on)|\s*[,.]|$)",
        r"(?:live|located|based)\s+in\s+([A-Z][a-zA-Z\s]+?)(?:\s*[,.]|$)"
    ]
    
    for pattern in location_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            params['starting_location'] = match.group(1).strip()
            break
    
    return params