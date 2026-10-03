import operator
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph
from langgraph.graph import START
from langgraph.graph import END

load_dotenv()


base_url = os.environ["OLLAMA_BASE_URL"]
api_key = os.environ["OLLAMA_API_KEY"]
model = os.environ["OLLAMA_MODEL"]

llm = ChatOllama(model=model, base_url=base_url,
client_kwargs={"headers": {"Authorization":
f"Bearer {api_key}"}}
)

# llm.num_ctx = 
# llm.temperature = 
# lm.seed = 

class State(TypedDict):
    msgs: Annotated[list[str], operator.add]

def updateMsgHistory(state: State):
    response = llm.invoke(state["msgs"])
    return{"msgs":[response.content]}

graph = StateGraph(State)
graph.add_node("updateMsgHistory", updateMsgHistory)

graph.add_edge(START, "updateMsgHistory")
graph.add_edge("updateMsgHistory", END)

app = graph.compile()


def llm_chatbot(user_input):
    return app.invoke({"msgs": [user_input]})


def main():

    print("LLM-based chatbot")
    print("'exit' to exit the chatbot")

    while True:
        user_input = input("User:")
        if(user_input == 'exit'):
            break
        response = llm_chatbot(user_input)
        print(f"Chatbot: {response['msgs'][-1]}\n")


if __name__ == "__main__":
    main()
