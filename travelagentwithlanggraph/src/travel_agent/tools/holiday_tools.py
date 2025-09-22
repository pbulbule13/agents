# src/travel_agent/tools/holiday_tools.py

import holidays
from datetime import datetime, timedelta
from langchain_core.tools import tool

@tool
def find_upcoming_long_weekend(weeks_in_future: int = 12) -> str:
    """
    Finds upcoming long weekends (3 or 4 days) in the US.
    This tool is useful for travel planning.
    
    Args:
        weeks_in_future: How many weeks from now to look for long weekends.
    """
    try:
        current_year = datetime.now().year
        # Fetch holidays for the current and next year to handle year-end holidays
        country_holidays = holidays.CountryHoliday('US', years=[current_year, current_year + 1])
        today = datetime.now().date()
        end_date = today + timedelta(weeks=weeks_in_future)
        
        long_weekends_found = []
        
        for holiday_date, holiday_name in country_holidays.items():
            if today <= holiday_date <= end_date:
                if holiday_date.weekday() in [4, 0]:  # Friday or Monday
                    long_weekends_found.append(
                        f"A 3-day weekend for {holiday_name} on {holiday_date.strftime('%B %d, %Y')}."
                    )
                elif holiday_date.weekday() in [3, 1]:  # Thursday or Tuesday
                    long_weekends_found.append(
                        f"A potential 4-day weekend around {holiday_name} on {holiday_date.strftime('%B %d, %Y')}."
                    )

        if long_weekends_found:
            return "Here are the upcoming long weekends I found: " + ", ".join(long_weekends_found)
        else:
            return "I couldn't find any long weekends in the next 12 weeks for the US."

    except Exception as e:
        return f"An error occurred while trying to find long weekends: {e}."