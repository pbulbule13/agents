# travel_agent/agents/itinerary_agent.py
import os
from langchain_core.messages import HumanMessage
from src.travel_agent.tools.holiday_tools import find_upcoming_long_weekend
from src.travel_agent.tools.geolocation_tools import get_coordinates, calculate_distance
from src.travel_agent.tools.accommodation_tools import search_vacation_rentals
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.pydantic_v1 import BaseModel, Field
from datetime import date, timedelta
from typing import List

# We define the input schema for the vacation planning agent
class VacationInput(BaseModel):
    num_guests: int = Field(..., description="The number of people in the group.")
    starting_location: str = Field(..., description="The starting city for the trip, e.g., 'Sunnyvale, CA'.")
    amenities: List[str] = Field(..., description="A list of desired amenities, e.g., ['game room', 'pool'].")

# A pre-defined list of potential destinations near Sunnyvale, CA
NEARBY_DESTINATIONS = {
    "Lake Tahoe": (39.0968, -120.0324),
    "Santa Cruz": (36.9741, -122.0308),
    "Napa Valley": (38.2975, -122.2868),
    "Yosemite National Park": (37.8651, -119.5383)
}

def plan_group_vacation(state: dict) -> dict:
    """
    Orchestrates the full vacation planning process.
    """
    messages = state.get("messages", [])
    user_query = messages[-1].content
    
    # Mock parameter extraction. In a real LLM-based agent, this would
    # be done by the model's tool-calling capability.
    # For this example, we'll assume a fixed user request.
    num_guests = 8
    starting_location = "Sunnyvale, CA"
    amenities = ["games", "pool table", "fire pit", "hot tub"]

    try:
        # Step 1: Find upcoming long weekends
        long_weekends = find_upcoming_long_weekend(num_months=18)
        
        # Step 2: Get starting location coordinates
        start_coords = get_coordinates(starting_location)
        
        # Step 3: Filter destinations by distance (100-150 miles)
        nearby_destinations = []
        for dest, coords in NEARBY_DESTINATIONS.items():
            distance = calculate_distance(start_coords, coords)
            if 100 <= distance <= 150:
                nearby_destinations.append(dest)
        
        if not nearby_destinations:
            return {"messages": messages + [HumanMessage(content=f"I couldn't find any suitable destinations within 100-150 miles of {starting_location}. Please try a different radius.")]}

        # Step 4: Search for accommodations for each valid weekend and destination
        planning_report = []
        for weekend in long_weekends:
            report_for_weekend = {
                "name": weekend['name'],
                "dates": f"From {weekend['start']} to {weekend['end']}",
                "accommodations": []
            }
            
            for destination in nearby_destinations:
                properties = search_vacation_rentals(
                    destination=destination,
                    check_in_date=str(weekend['start']),
                    check_out_date=str(weekend['end']),
                    num_guests=num_guests,
                    amenities=amenities
                )
                
                if properties:
                    report_for_weekend['accommodations'].append({
                        "destination": destination,
                        "properties": properties
                    })
            
            if report_for_weekend['accommodations']:
                planning_report.append(report_for_weekend)

        # Step 5: Format the final response
        if not planning_report:
            final_response = "I couldn't find any suitable accommodations with your desired amenities for the upcoming long weekends."
        else:
            final_response = "Here is your group vacation plan for the upcoming long weekends:\n\n"
            for report in planning_report:
                final_response += f"### {report['name']} ({report['dates']})\n"
                for accom_info in report['accommodations']:
                    final_response += f"#### Destination: {accom_info['destination']}\n"
                    for prop in accom_info['properties']:
                        final_response += (
                            f"- **{prop['name']}**\n"
                            f"  - Summary: {prop['summary']}\n"
                            f"  - URL: {prop['url']}\n"
                            f"  - Amenities: {', '.join(prop.get('amenities', []))}\n"
                        )
                final_response += "\n---\n"
        
        return {"messages": messages + [HumanMessage(content=final_response)]}
        
    except Exception as e:
        error_response = f"I encountered an error while planning your getaway: {str(e)}."
        return {"messages": messages + [HumanMessage(content=error_response)]}