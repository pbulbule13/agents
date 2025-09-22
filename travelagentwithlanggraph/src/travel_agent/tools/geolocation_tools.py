# src/travel_agent/tools/geolocation_tools.py
import os
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from langchain_core.tools import tool
from typing import Dict, Any

# Initialize a geolocator service (Nominatim is a free, simple option)
geolocator = Nominatim(user_agent="travel_planner_app")

@tool
def get_coordinates(location: str) -> Dict[str, Any]:
    """
    Finds the geographical coordinates (latitude and longitude) for a given location.
    This tool is useful for subsequent distance calculations or location-based searches.
    """
    try:
        location_data = geolocator.geocode(location)
        if location_data:
            return {
                "latitude": location_data.latitude,
                "longitude": location_data.longitude,
                "address": location_data.address
            }
        else:
            return {"error": f"Could not find coordinates for {location}."}
    except Exception as e:
        return {"error": f"An error occurred while geocoding {location}: {e}."}


@tool
def calculate_distance(location1: str, location2: str) -> str:
    """
    Calculates the straight-line distance between two locations in miles.
    This tool uses geocoding to find coordinates first.
    """
    try:
        coords1 = get_coordinates(location1)
        coords2 = get_coordinates(location2)

        if "error" in coords1:
            return coords1["error"]
        if "error" in coords2:
            return coords2["error"]

        point1 = (coords1["latitude"], coords1["longitude"])
        point2 = (coords2["latitude"], coords2["longitude"])
        
        distance_miles = geodesic(point1, point2).miles
        return f"The distance between {location1} and {location2} is approximately {distance_miles:.2f} miles."
    
    except Exception as e:
        return f"An error occurred while calculating the distance: {e}."