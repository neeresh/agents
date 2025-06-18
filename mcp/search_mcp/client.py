import asyncio
import os
import json

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from dotenv import load_dotenv

load_dotenv()

PAT = os.environ.get("CLARIFAI_PAT")

# MCP Server Configuration
url = "https://api.clarifai.com/v2/ext/mcp/v1/users/nperla/apps/mcp-examples/models/blog_writing_search_mcp"
transport = StreamableHttpTransport(url=url, headers={"Authorization": "Bearer " + PAT})

async def main():
    print("=== SerpAPI MCP Server ===\n")
    
    async with Client(transport=transport) as client:
        print("Available Tools: ")
        try:
            tools = await client.list_tools()
            for tool in tools:
                print(f"- {tool.name}: {tool.description}")
        
        except Exception as e:
            print(f"Error listing tools: {e}")
            return
    
    # Multi-Engine Search
    print("Testing: multi_engine_search...")
    try:
        result = await client.call_tool(
            "multi_engine_search",
            {"query": "AI in healthcare", "engine": "google", "location": "United States"}
            )
        
        response_data = json.loads(result[0].text)
        if isinstance(response_data, list):
            print("Top 5 links:")
            for link in response_data:
                print(f"- {link}")
        else:
            print("Unexpected response format: not a list")
    
    except Exception as e:
        print(f"Error calling multi_engine_search: {e}")
        
    print("\n" + "="*50 + "\n")
    
    # 3. Extract content from links
    print("Testing: extract_web_content_from_links...")
    try:
        links = [
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC8285156/",
            "https://www.foreseemed.com/artificial-intelligence-in-healthcare",
            "https://news.harvard.edu/gazette/story/2025/03/how-ai-is-transforming-medicine-healthcare/"
        ]
        result = await client.call_tool("extract_web_content_from_links", {"urls": links})
        response_data = json.loads(result[0].text)
        for url, content in response_data.items():
            print(f"\nURL: {url}\nExtracted Content (first 500 chars):\n{content[:500]}")
            
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "="*50 + "\n")

    # 4. Keyword research
    print("Testing: keyword_research...")
    try:
        result = await client.call_tool(
            "keyword_research",
            {"topic": "home automation"}
        )
        response_data = json.loads(result[0].text)
        print("Keyword research results:")
        for item in response_data:
            print(f"- Keyword: {item.get('keyword')}, Popularity: {item.get('relative_popularity_score')}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())



