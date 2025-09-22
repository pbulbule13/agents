# Replace your router.py content in src/travel_agent/agents/router.py

def route_to_agent(state):
    """
    Enhanced router for weekend trip planning focus.
    Routes most requests to itinerary agent unless specifically about flights.
    """
    
    messages = state.get("messages", [])
    if not messages:
        return "itenary_agent"
    
    last_message = messages[-1].content.lower()
    
    # Flight-specific routing (only for explicit flight requests)
    flight_keywords = [
        "flight", "fly", "airline", "airport", "departure", "arrival", 
        "book flight", "flight booking", "plane ticket", "airfare"
    ]
    
    # Only route to flight agent for explicit flight requests
    if any(keyword in last_message for keyword in flight_keywords):
        # But still route to itinerary for general trip planning with flights
        planning_context = [
            "plan", "trip", "itinerary", "weekend", "vacation", "getaway", 
            "visit", "travel to", "going to", "schedule", "agenda"
        ]
        
        has_planning_context = any(context in last_message for context in planning_context)
        
        if not has_planning_context:
            return "flight_agent"
    
    # Everything else goes to itinerary agent for comprehensive weekend planning
    # This includes accommodation, activities, restaurants, scheduling, etc.
    return "itenary_agent"


def create_router():
    """
    Create a router optimized for weekend trip planning workflow.
    """
    
    def router_node(state):
        """
        Router node that analyzes requests and routes appropriately.
        Most requests go to itinerary agent for comprehensive weekend planning.
        """
        messages = state.get("messages", [])
        
        if not messages:
            return state
        
        last_message = messages[-1].content
        print(f"🧭 Weekend Trip Router analyzing: '{last_message[:50]}...'")
        
        # Determine routing
        agent_choice = route_to_agent(state)
        print(f"🎯 Routing to: {agent_choice}")
        
        # Add some context about the routing decision
        if agent_choice == "itenary_agent":
            print("📋 → Comprehensive weekend trip planning mode")
        elif agent_choice == "flight_agent":
            print("✈️ → Flight-specific assistance mode")
        
        return state
    
    return router_node