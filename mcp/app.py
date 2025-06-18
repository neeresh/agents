import sys
import streamlit as st
import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import MCPServerAdapter

from dotenv import load_dotenv

load_dotenv()

CLARIFAI_PAT = os.getenv("CLARIFAI_PAT")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

if not CLARIFAI_PAT:
    st.error("Please set CLARIFAI_PAT environment variable")
    st.stop()

if not SERPER_API_KEY:
    st.error("Please set SERPER_API_KEY environment variable")
    st.stop()

# Configure Clarifai LLM
clarifai_llm = LLM(
    model="openai/openai/chat-completion/models/gpt-4o",
    api_key=CLARIFAI_PAT,
    base_url="https://api.clarifai.com/v2/ext/openai/v1"
)

# MCP Server Configuration
USER_ID = "nperla"
APP_ID = "mcp-examples"
MODEL_ID = "blog_writing_search_mcp"

server_params = {
    "url": f"https://api.clarifai.com/v2/ext/mcp/v1/users/{USER_ID}/apps/{APP_ID}/models/{MODEL_ID}",
    "headers": {"Authorization": "Bearer " + CLARIFAI_PAT},
    "transport": "streamable-http"
}

# Streamlit App
def main():
    st.set_page_config(page_title="AI Blog Writing Agent", page_icon="📝", layout="wide")
    st.title("📝 AI Blog Writing Agent")
    st.markdown("<h2 style='text-align: center; color: #2E86C1;'><strong>Powered by Clarifai, CrewAI & a Custom SerpAPI MCP Server</strong></h2>", unsafe_allow_html=True)

    st.markdown("""
    **How it works:**
    - ✅ **Planner Agent**: Researches top articles and extracts SEO keywords.
    - ✍️ **Writer Agent**: Writes a full blog post using the research and outline.
    - 🔎 **Editor Agent**: Polishes the final post and formats it in markdown.
    """)

    topic = st.text_input(
        "Enter your blog topic:",
        placeholder="e.g., The Future of Quantum Computing",
        help="Be specific for better results"
    )

    generate_button = st.button("🚀 Generate Blog", type="primary")

    if generate_button:
        if not topic.strip():
            st.error("Please enter a topic for the blog post.")
        else:
            with st.spinner(f"Running agents on: '{topic}'..."):
                try:
                    with MCPServerAdapter(server_params) as mcp_tools:
                        st.info(f"✅ Connected to MCP Server. Tools: {[tool.name for tool in mcp_tools]}")

                        # Agents
                        planner = Agent(
                            role="SEO Researcher and Content Planner",
                            goal="Extract key insights, find SEO keywords, and outline the blog.",
                            backstory="You research top articles and produce outlines optimized for engagement and SEO.",
                            tools=mcp_tools,
                            verbose=True,
                            llm=clarifai_llm,
                            allow_delegation=False
                        )

                        writer = Agent(
                            role="Blog Post Writer",
                            goal="Create a detailed, high-quality blog post using the research and outline.",
                            backstory="You are a writer who specializes in transforming outlines into compelling blog posts.",
                            verbose=True,
                            llm=clarifai_llm,
                            allow_delegation=False
                        )

                        editor = Agent(
                            role="Blog Editor and Formatter",
                            goal="Edit the blog post, correct grammar, and format it in markdown.",
                            backstory="You ensure every blog is well-written, polished, and correctly formatted for publishing.",
                            verbose=True,
                            llm=clarifai_llm,
                            allow_delegation=False
                        )

                        # Tasks
                        plan_task = Task(
                            description=f"""
                                For the topic "{topic}":

                                1. Use `multi_engine_search` to find 5 recent, relevant articles.
                                2. Extract content using `extract_web_content_from_links`.
                                3. Use `keyword_research` to find SEO keywords.
                                4. Summarize key findings and generate a structured outline.

                                The outline should include:
                                - Title
                                - Introduction
                                - 3-4 section headings with bullet points
                                - Conclusion
                                """,
                            expected_output="A blog outline with insights, 5-10 SEO keywords, and detailed structure.",
                            agent=planner
                        )

                        write_task = Task(
                            description=f"""
                                Using the outline and research for "{topic}", write a complete blog post with:

                                - At least 5–6 paragraphs
                                - Natural integration of SEO keywords
                                - Engaging and informative tone
                                - Clean markdown format

                                Use examples and factual support where possible.
                                """,
                            expected_output="Full markdown blog post draft, ready for editing.",
                            agent=writer,
                            context=[plan_task]
                        )

                        edit_task = Task(
                            description=f"""
                                Edit the blog post for "{topic}":

                                - Fix grammar and clarity issues
                                - Enhance tone, transitions, and flow
                                - Ensure SEO keywords are present naturally
                                - Format properly in markdown:
                                    - Use `#` for title
                                    - `##` for section headers
                                    - Paragraph spacing and bullet points
                                    - Bold key phrases if needed

                                Return the final polished markdown content.
                                """,
                            expected_output="Final markdown blog post, ready for publishing.",
                            agent=editor,
                            context=[write_task]
                        )

                        # Run the Crew
                        crew = Crew(
                            agents=[planner, writer, editor],
                            tasks=[plan_task, write_task, edit_task],
                            process=Process.sequential,
                            verbose=1
                        )
                        
                        result = crew.kickoff()

                        final_output = result.output if hasattr(result, "output") else str(result)

                        st.success("✅ Blog post generated successfully!")
                        st.markdown("---")
                        st.markdown(final_output, unsafe_allow_html=False)

                        st.download_button(
                            label="📥 Download as Markdown",
                            data=final_output,
                            file_name=f"{topic.replace(' ', '_').lower()}_blog.md",
                            mime="text/markdown"
                        )

                except Exception as e:
                    st.error(f"An error occurred: {e}")

    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("**MCP Server:**")
        st.code(f"USER_ID: {USER_ID}\nAPP_ID: {APP_ID}\nMODEL_ID: {MODEL_ID}", language="text")
        st.markdown("**LLM Config:**")
        st.markdown("- Model: `gpt-4o` via Clarifai")
        st.markdown("- Base URL: `https://api.clarifai.com`")

        st.header("🛠️ Features")
        st.markdown("- Research via search + content extraction")
        st.markdown("- Keyword generation for SEO")
        st.markdown("- Plan → Write → Edit flow with agents")

        st.warning("⚠️ Keep your API keys secure. Ensure the MCP server is live on Clarifai.")

if __name__ == "__main__":
    main()

