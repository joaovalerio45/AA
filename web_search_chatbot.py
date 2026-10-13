import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

base_url = os.environ["OLLAMA_BASE_URL"]
api_key = os.environ["OLLAMA_API_KEY"]
model = os.environ["OLLAMA_MODEL"]

llm = ChatOllama(
    model=model,
    base_url=base_url,
    client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}},
)

web_search = TavilySearch(max_results=3)
tools = [web_search]
llm = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def updateMsgHistory(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph = StateGraph(State)
graph.add_node("updateMsgHistory", updateMsgHistory)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "updateMsgHistory")
graph.add_conditional_edges("updateMsgHistory", tools_condition)
graph.add_edge("tools", "updateMsgHistory")

app = graph.compile()


def llm_chatbot(user_input):
    return app.invoke({"messages": [("user", user_input)]})


def main():
    print("LLM-based chatbot with Web Search")
    print("'exit' to exit the chatbot\n")

    while True:
        user_input = input("User: ")
        if user_input.strip().lower() == "exit":
            break
        response = llm_chatbot(user_input)
        print(f"Chatbot: {response['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
