import streamlit as st
import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

load_dotenv()

CLARIFAI_API = os.getenv("CLARIFAI_API")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

clarifai_llm = LLM(
    model="openai/gcp/generate/models/gemini-2_5-pro",
    api_key=CLARIFAI_API,
    base_url="https://api.clarifai.com/v2/ext/openai/v1",
)

search_tool = SerperDevTool()


# Define Agents
researcher = Agent(
    role="Senior Research Analyst",
    goal=(
        "Systematically identify, validate, and synthesize the most recent, "
        "high-impact findings and developments on a given technical topic, "
        "and produce a clear, well-structured briefing with source attributions."
    ),
    backstory="""
        You are a Senior Research Analyst at a leading technology think tank, renowned for your rigorous
        methodology and attention to detail. You excel at:

        • Scoping the project: defining subtopics, keywords, and questions that uncover hidden insights.  
        • Source vetting: prioritizing peer-reviewed papers, reputable industry reports, and authoritative blogs.  
        • Synthesis: distilling complex material into concise bullet points, annotated with citations and confidence levels.  
        • Collaboration: asking clarifying questions when the research scope is ambiguous or too broad.
        """.strip(),
    tools=[search_tool],
    verbose=True,
    allow_delegation=False,
    llm=clarifai_llm
)

writer = Agent(
    role="Tech Content Strategist & Copywriter",
    goal=(
        "Transform research insights into an engaging, SEO-optimized blog post "
        "that communicates complex concepts to a tech-savvy audience, with a clear narrative flow."
    ),
    backstory="""
        You are a seasoned Tech Content Strategist and Copywriter, published across top industry outlets.
        Your strengths include:

        • Audience profiling: adapting tone, terminology, and depth for developers, researchers, or decision-makers.  
        • Storytelling: crafting a compelling introduction, logical progression of ideas, and memorable takeaways.  
        • SEO best practices: integrating keywords naturally, writing persuasive headings, and optimizing for readability.  
        • Revision: incorporating feedback, refining clarity, and ensuring factual accuracy via collaboration with the researcher.
        """.strip(),
    verbose=True,
    allow_delegation=True,
    llm=clarifai_llm
)

def create_tasks(topic):
    
    research_task = Task(
        description=(
            f"Conduct a structured, in-depth exploration of “{topic}”:\n"
            "1. Define key subtopics and research questions.\n"
            "2. Source and vet high-quality materials (peer-reviewed papers, industry reports, authoritative blogs).\n"
            "3. Synthesize findings into concise bullet points with source citations and confidence annotations.\n"
            "4. Highlight emerging trends, breakthrough technologies, leading experts, and potential industry impacts."
        ),
        expected_output=(
            "A comprehensive research briefing: a bullet-point summary with clear source attributions "
            "and confidence ratings."
        ),
        agent=researcher
    )

    writing_task = Task(
        description=(
            f"Using the research briefing on “{topic}”, craft an engaging, SEO-optimized blog post:\n"
            "1. Create a compelling title (use `#`).\n"
            "2. Organize content with section headers (use `##`) for Introduction, Body, and Conclusion.\n"
            "3. Maintain a logical narrative flow and approachable, human tone.\n"
            "4. Explain any technical terms and avoid unexplained jargon.\n"
            "5. Use bullet points to clarify lists.\n"
            "6. Emphasize key insights with **bold** text.\n"
            "7. Output strictly as Markdown (no code blocks or triple backticks)."
        ),
        expected_output=(
            "A well-crafted Markdown blog post (4–6 paragraphs) with title, headings, bullet points, and emphasis."
        ),
        agent=writer,
        context=[research_task]
    )

    return research_task, writing_task

def run_blog_generation(topic):
    research_task, writing_task = create_tasks(topic)
    crew = Crew(agents=[researcher, writer],
                tasks=[research_task, writing_task],
                process=Process.sequential,
                verbose=True)
    result = crew.kickoff()
    
    return result

def main():
    st.set_page_config(page_title="AI Blog Writer", page_icon=":robot:",
                       layout="centered")
    st.markdown("_Powered by Clarifai Gemini 2.5 Pro & CrewAI Agents_")
    st.write("---")
    
    st.header("How It Works")
    st.markdown("""
                
    1. 🔍 **Researcher Agent**  
       - Defines subtopics and research questions  
       - Sources and vets high-quality materials  
       - Synthesizes findings into concise, cited bullet points  
       
    2. ✍️ **Writer Agent**  
       - Crafts an SEO-optimized title and headings  
       - Converts research into a human-friendly narrative  
       - Emphasizes key insights with **bold** text  
       - Outputs clean Markdown (no code blocks)  
    """)
    
    st.write("---")
    
    # Input Section
    with st.container():
        topic = st.text_input(
            "Enter you blog topic:",
            placeholder="e.g., 'e.g., Advances in Robotic Vision'",
            help="Be specific for better results")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            generate_button = st.button("Generate Blog", type="primary")
            
    if generate_button:
        if not topic.strip():
            st.error("Please enter a topic for the blog post.")
        else:
            with st.spinner(f"AI agents are working on: '{topic}'..."):
                try:
                    result = run_blog_generation(topic=topic)
                    
                    result = result
                    st.success("Blog post generated successfully!")
                    st.markdown(result)
                    
                    st.download_button(
                        label="Download Blog Post",
                        data=result,
                        file_name=f"{topic.replace(' ', '_')}_blog_post.md",
                        mime="text/markdown"
                    )
                
                except Exception as e:
                    st.error(f"And error occurred: {str(e)}")
    
    # Sidebar Information
    with st.sidebar:
        st.header("Information")
        
        st.markdown("**Environment Variables Required:**")
        st.code("CLARIFAI_PAT=your_clarifai_personal_access_token")
        st.code("SERPER_API_KEY=your_serper_dev_api_key")
        
        st.markdown("**Current Configuration:**")
        st.markdown(f"- **Model:** `gcp/generate/models/gemini-2_5-pro`")
        st.markdown(f"- **API Base:** `api.clarifai.com`")
        
        st.markdown("**Features:**")
        st.markdown("- Real-time web research")
        st.markdown("- AI-powered content writing")
        st.markdown("- Markdown formatted output")
        st.markdown("- Download capability")
        
        st.warning("⚠️ Keep your API keys secure and never commit them to version control.")
        
if __name__ == "__main__":
    
    # Topic: Unified Multimodal Transformers: The Next Frontier in Embodied AI
    main()
    