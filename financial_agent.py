from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
import openai
from dotenv import load_dotenv


load_dotenv()
openai.api_key= os.getenv('OPEN_AI_API_KEY')
# Create a new agent

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


multi_ai_agent = Agent(
    team=[web_search_agent,financial_agent],
    instructions=["Always include sources","Use table to display financial data"],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response("Summarize analyst recomendation and share the latest new for NVidia",stream=True)
