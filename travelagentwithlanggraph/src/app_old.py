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
# Since app.py is in src/ and travel_agent is also in src/, 
# we just need to add the current directory (src/) to the path
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

st.success("✅ Path setup completed")

# Debug info (remove later)
with st.expander("Debug Info", expanded=False):
    st.write(f"Current directory: {current_dir}")
    st.write(f"travel_agent exists: {os.path.exists(os.path.join(current_dir, 'travel_agent'))}")
    st.write(f"graph.py exists: {os.path.exists(os.path.join(current_dir, 'travel_agent', 'graph.py'))}")

# --- IMPORTS ---
try:
    # Simple import since both app.py and travel_agent are in src/
    from travel_agent.graph import build_travel_graph
    from langchain_core.messages import HumanMessage
    st.success("✅ Imports completed")
except Exception as e:
    st.error(f"❌ Import failed: {e}")
    st.write("**Troubleshooting:**")
    st.write("Make sure all required packages are installed:")
    st.code("pip install langchain langgraph langchain-community langchain-google-genai tavily-python")
    st.stop()

# --- API KEY HANDLING ---
def check_api_keys():
    """Checks if all required API keys are in st.secrets."""
    required_keys = ["GOOGLE_API_KEY", "TAVILY_API_KEY", "SERPAPI_API_KEY"]
    missing_keys = [key for key in required_keys if key not in st.secrets]
    
    if missing_keys:
        st.error(
            f"❌ Missing API keys: {', '.join(missing_keys)}\n\n"
            f"Please add them to your `.streamlit/secrets.toml` file"
        )
        st.info(
            "Create `.streamlit/secrets.toml` in your project root with:\n\n"
            "```toml\n"
            "GOOGLE_API_KEY = \"your-key-here\"\n"
            "TAVILY_API_KEY = \"your-key-here\"\n"
            "SERPAPI_API_KEY = \"your-key-here\"\n"
            "```"
        )
        st.stop()
    else:
        st.success("✅ All API keys found")

# Check API keys
check_api_keys()

# --- ENVIRONMENT SETUP ---
try:
    # Set environment variables
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"] 
    os.environ["SERPAPI_API_KEY"] = st.secrets["SERPAPI_API_KEY"]
    
    # Configure Google AI
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    
    st.success("✅ Environment configured")
    
except Exception as e:
    st.error(f"❌ Environment setup failed: {e}")
    st.stop()

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# --- GRAPH INITIALIZATION ---
if "travel_planner" not in st.session_state:
    try:
        with st.spinner("🤖 Initializing travel planning agents... This may take a moment."):
            st.session_state.travel_planner = build_travel_graph()
        st.success("✅ Travel planner ready!")
    except Exception as e:
        st.error(f"❌ Failed to initialize travel planner: {e}")
        st.write("**Error details:**")
        st.code(str(e))
        
        # Common troubleshooting
        st.write("**Common solutions:**")
        st.write("1. Make sure all API keys are valid")
        st.write("2. Check internet connection")
        st.write("3. Verify all packages are installed correctly")
        st.stop()

# --- CHAT INTERFACE ---
st.markdown("---")
st.subheader("💬 Chat with your Travel Assistant")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("What travel destination would you like to explore?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        try:
            with st.spinner("🧠 Planning your trip..."):
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