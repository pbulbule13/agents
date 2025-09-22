# src/travel_agent/tools/accommodation_tools.py

import requests
import os
from langchain_core.tools import tool
from scrapfly import ScrapflyClient, ScrapeConfig
from bs4 import BeautifulSoup

@tool
def search_vacation_rentals(
    destination: str,
    check_in_date: str,
    check_out_date: str,
    number_of_guests: int = 2,
    price_range: str = "any",
    platform: str = "any"
) -> str:
    """
    Searches for vacation rentals on platforms like Airbnb or VRBO.
    This tool requires a destination, check-in date, and check-out date.
    It can optionally filter by number of guests, price range, and specific platforms.
    """
    RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
    URL = "https://example-rental-api.p.rapidapi.com/search"  # Replace with actual API URL
    HOST = "example-rental-api.p.rapidapi.com"  # Replace with actual API Host

    querystring = {
        "query": f"{destination} vacation rentals",
        "checkin": check_in_date,
        "checkout": check_out_date,
        "adults": number_of_guests,
        "sort_by": "best_match"
    }

    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": HOST
    }

    try:
        response = requests.request("GET", URL, headers=headers, params=querystring)
        response.raise_for_status()
        data = response.json()

        if not data.get('results'):
            return "Sorry, I couldn't find any listings for your request."

        listings = data['results'][:3] # Get top 3 results
        summary = "I've found some excellent options for you:\n\n"
        for listing in listings:
            summary += f"- **{listing['name']}**\n"
            summary += f"  - Location: {listing.get('address', 'N/A')}\n"
            summary += f"  - Price: {listing.get('price', 'N/A')}\n"
            summary += f"  - URL: {listing.get('url', 'N/A')}\n\n"
        
        return summary

    except requests.exceptions.RequestException as e:
        return f"An error occurred while searching for accommodations: {str(e)}"

@tool
def scrape_listing_page(url: str) -> str:
    """
    Scrapes a single vacation rental listing page for details.
    This tool requires a valid URL to a listing.
    """
    SCRAPFLY_KEY = os.environ.get("SCRAPFLY_API_KEY")
    client = ScrapflyClient(key=SCRAPFLY_KEY)
    
    try:
        response = client.scrape(
            ScrapeConfig(url=url, render_js=True, country='us')
        )
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # This is example logic to find price and number of beds
        price_tag = soup.find('span', {'data-testid': 'price_amount'})
        beds_tag = soup.find('div', class_='beds_number')
        
        price = price_tag.text if price_tag else "Price not found"
        beds = beds_tag.text if beds_tag else "Beds not found"
        
        return f"I successfully scraped the page. Details found:\nPrice: {price}\nBeds: {beds}"
        
    except Exception as e:
        return f"An error occurred while scraping the page: {str(e)}"