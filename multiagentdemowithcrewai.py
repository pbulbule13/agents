import os
from dotenv import load_dotenv
from crewai import Agent , Task , Crew
from crewai_tools import SerperDevTool  # Fixed import
from langchain_openai import ChatOpenAI


load_dotenv()

serper_api_key = os.getenv("SERPER_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

search_tool = SerperDevTool()


def create_research_agent():

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    return Agent(
        role="Research Specialist",
        goal="Conduct thorough research on given topics",
        backstory="You are an experienced researcher with expertise in finding and synthesizing information from various sources",
        verbose=True,
        allow_delegation=True,
        tools=[search_tool],
        llm=llm,
    )

def create_writer_agent():

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    return Agent(
        role="Research Specialist",
        goal="Conduct thorough research on given topics",
        backstory="You are an experienced researcher with expertise in finding and synthesizing information from various sources",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
    )


def create_research_task(agent, topic):
    return Task(
        description=f"Research the following topic and provide a comprehensive summary: {topic}",
        agent=agent,
        expected_output="A detailed summary of the research findings, including key points and insights related to the topic"
    )

def create_write_task(agent, topic):
    return Task(
        description=f"Research the following topic and provide a comprehensive summary: {topic}",
        agent=agent,
        expected_output="A detailed summary of the research findings, including key points and insights related to the topic",
        output_file="newblog.md",
    )



def run_research(topic):
    researchagent = create_research_agent()
    taskresearch= create_research_task(researchagent, topic)

    writeragent = create_writer_agent()
    taskwriter= create_write_task(writeragent, topic)

    crew = Crew(agents=[researchagent,writeragent],tasks=[taskresearch,taskwriter])
    result = crew.kickoff()
    return result



if __name__ == "__main__":
    print("Welcome to the Research Agent!")
    topic = input("Enter the research topic: ")
    result = run_research(topic)

#     # Forming the tech-focused crew with enhanced configurations
# crew = Crew(
#     agents=[researcher, writer],
#     tasks=[research_task, write_task],
#     process=Process.sequential  # Optional: Sequential task execution is default
# )
    print("Research Result:")
    print(result)

