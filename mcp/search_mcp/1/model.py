"""
Main file for MCP Server Logic
"""

import os

from typing import Annotated, Dict, Any, List
from pydantic import Field
from serpapi import GoogleSearch
from clarifai.runners.models.mcp_class import MCPModelClass
from fastmcp import FastMCP
from newspaper import Article

from dotenv import load_dotenv

load_dotenv()

server = FastMCP("blog_writing_search_mcp")
SERPAPI_API_KEY = os.getenv("SERPER_API_KEY")


@server.tool("multi_engine_search", description="Query a search engine and return the top 5 blog/article links based on a search query.")
def multi_engine_search(query: Annotated[str, Field(description="Search query")],
                        engine: Annotated[str, Field(description="Search engine to use (e.g., 'google).")] = "google",
                        location: Annotated[str, Field(description="Geographic location for the search.")] = "United States",
                        device: Annotated[str, Field(description="Device type for the search (e.g., 'desktop', 'mobile').")] = "desktop",
                        ) -> List[str]:
    
    params = {"api_key": SERPAPI_API_KEY, "engine": engine, "q": query, 
              "location": location, "device": device,}
    
    search = GoogleSearch(params)
    results = search.get_dict()
    
    links = []
    for result in results.get("organic_results", [])[:5]:
        link = result.get("link")
        if link:
            links.append(link)
    
    return links
    

@server.tool("extract_web_content_from_links",
             description="Extracts main article content from a list of blog or article URLs using newspaper3k.")    
def extract_web_content_from_links(
    urls: Annotated[List[str], Field(description="List of blog/article URLs to extract content from.")]) -> Dict[str, str]:
    
    extracted = {}
    
    for url in urls:
        try:
            article = Article(url)
            article.download()
            article.parse()
            extracted[url] = article.text[:1000]  # Extract first 1000 characters of the article text
        
        except Exception as e:
            extracted[url] = f"Error extracting content: {str(e)}"
        
    return extracted


@server.tool(
    "keyword_research",
    description="Automate keyword research to find high-potential keywords based on a topic, using autocomplete and trends."
)
def keyword_research(
    topic: Annotated[str, Field(description="Blog topic to research keywords for.")]) -> List[Dict[str, Any]]:
    
    autocomplete_params = {"api_key": SERPAPI_API_KEY, "engine": "google_autocomplete", 
                           "q": topic,}
    search = GoogleSearch(autocomplete_params)
    
    autocomplete_results = search.get_dict()
    suggestions = [item['value'] for item in autocomplete_results.get('suggestions', [])[:5]]
    
    if not suggestions:
        return [{"error": "Could not fetch keyword suggestions."}]
    
    trends_params = {"api_key": SERPAPI_API_KEY, "engine": "google_trends",
                     "q": ", ".join(suggestions), "data_type": "TIMESERIES"}
    
    search = GoogleSearch(trends_params)
    trends_results = search.get_dict()
    
    keyword_data = []
    if "interest_over_time" in trends_results:
        timeline_data = trends_results["interest_over_time"].get("timeline_data", [])
        for i, keyword in enumerate(suggestions):
            last_value = timeline_data[-1]['values'][i].get('value') if timeline_data else "N/A"
            keyword_data.append({
                "keyword": keyword,
                "relative_popularity_score": last_value
            })
    else:
        for keyword in suggestions:
            keyword_data.append({
                "keyword": keyword,
                "relative_popularity_score": "N/A"
            })
    
    return keyword_data


class MyModelClass(MCPModelClass):
    def get_server(self) -> FastMCP:
        return server
    