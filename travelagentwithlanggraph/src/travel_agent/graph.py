from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from src.travel_agent.state import TravelPlannerState
from src.travel_agent.agents.agents import itinerary_agent_node, flight_agent_node, hotel_agent_node
from src.travel_agent.agents.router import create_router, route_to_agent

def build_travel_graph():
    checkpointer = InMemorySaver()

    # FIX: The workflow variable needs to be created first!
    workflow = StateGraph(TravelPlannerState)

    # Add all nodes to the graph
    workflow.add_node("router", create_router())
    workflow.add_node("flight_agent", flight_agent_node)
    workflow.add_node("hotel_agent", hotel_agent_node)
    workflow.add_node("itenary_agent", itinerary_agent_node)

    # Set entry point
    workflow.set_entry_point("router")

    # Add conditional edges from router to appropriate agent
    workflow.add_conditional_edges(
        "router",
        route_to_agent,
        {
            "flight_agent": "flight_agent",
            "hotel_agent": "hotel_agent",
            "itenary_agent": "itenary_agent"
        }
    )

    # Add edges from each agent back to END
    workflow.add_edge("flight_agent", END)
    workflow.add_edge("hotel_agent", END)
    workflow.add_edge("itenary_agent", END)

    # Compile the graph
    travel_planner = workflow.compile(checkpointer=checkpointer)
    print("✅ Travel Planning Graph built successfully!")
    return travel_planner