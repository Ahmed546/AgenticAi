import openai
from phi.agent import Agent
import phi.api
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.groq import Groq
from dotenv import load_dotenv

import os
import phi
from phi.playground import Playground,serve_playground_app

#load environment variable from .env file
load_dotenv()

phi.api=os.getenv("PHI_API_KEY")


web_search_agent = Agent(
    name="Web Search Agent",
    role="search web from information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include the sources"],
    show_tool_calls=True,
    markdown=True,

)


##financial agent
financial_agent = Agent(
    name="Financial Agent",
    role="analyze financial data",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,company_news=True)],
    instructions=["Use tables to display financial data"],
    show_tool_calls=True,
    markdown=True,
)

app= Playground(agents=[financial_agent,web_search_agent]).get_app()

if __name__ == '__main__':
    
    serve_playground_app("playground:app",reload=True)