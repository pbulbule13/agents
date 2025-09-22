import streamlit as st
import os
import sys
import uuid

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="Multi-Agent Travel Assistant",
    layout="wide"
)

st.title("✈️ Multi-Agent Travel Assistant")

# --- PATH SETUP ---
project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- IMPORTS ---
try:
    from src.travel_agent.graph import build_travel_graph
    from langchain_core.messages import HumanMessage
except Exception as e:
    st.error(f"❌ Setup error: {e}")
    st.info("Please make sure all required packages are installed and your API keys are configured.")
    st.stop()

# --- API KEY HANDLING ---
def check_api_keys():
    required_keys = ["GOOGLE_API_KEY", "TAVILY_API_KEY", "SERPAPI_API_KEY", "EURI_API_KEY"]
    missing_keys = [key for key in required_keys if key not in st.secrets]
    
    if missing_keys:
        st.error(f"❌ Missing API keys: {', '.join(missing_keys)}")
        st.info(
            "Create `.streamlit/secrets.toml` with:\n\n"
            "```toml\n"
            "GOOGLE_API_KEY = \"your-key-here\"\n"
            "TAVILY_API_KEY = \"your-key-here\"\n"
            "SERPAPI_API_KEY = \"your-key-here\"\n"
            "EURI_API_KEY = \"your-key-here\"\n"
            "```"
        )
        st.stop()

check_api_keys()

# --- ENVIRONMENT SETUP ---
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"] 
    os.environ["SERPAPI_API_KEY"] = st.secrets["SERPAPI_API_KEY"]
    os.environ["EURI_API_KEY"] = st.secrets["EURI_API_KEY"]
    
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except Exception as e:
    st.error(f"❌ Configuration error: {e}")
    st.stop()

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# --- GRAPH INITIALIZATION ---
if "travel_planner" not in st.session_state:
    try:
        with st.spinner("🤖 Initializing your travel planning assistant..."):
            st.session_state.travel_planner = build_travel_graph()
        st.success("✅ Ready to plan your next adventure!")
    except Exception as e:
        st.error(f"❌ Initialization failed: {e}")
        st.stop()

# --- CHAT INTERFACE ---
st.markdown("---")
st.subheader("💬 Plan Your Perfect Weekend Getaway")

# Show a helpful prompt if no messages
if not st.session_state.messages:
    st.info("👋 **Ready to plan an amazing weekend trip?**\n\n"
           "Try asking me:\n"
           "• 'Plan a weekend trip to Napa Valley for 4 people'\n"
           "• 'I want to visit Lake Tahoe for skiing, January 14-16'\n" 
           "• 'Long weekend in San Francisco for foodies'")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Where would you like to go for your next weekend adventure?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        try:
            with st.spinner("🧠 Creating your perfect itinerary..."):
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = st.session_state.travel_planner.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config
                )
                response = result["messages"][-1].content
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})