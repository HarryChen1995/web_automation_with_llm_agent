from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from web_automation import google_search, driver, go_to_web_page

embeddings = OllamaEmbeddings(model="llama3.2")

llm = ChatOllama(model="llama3.2", temperature= 100)





from langchain_core.tools import tool


@tool
def search_internet(query: str):
    """serch internet with a query."""
    urls = google_search(driver, query)
    return '\n'.join(urls)


@tool
def summarize_web_content_from_link(link: str):
    """summarize web content from link."""
    pages = go_to_web_page(driver, link)
    return "".join([page.page_content for page in pages])

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([search_internet, summarize_web_content_from_link])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
search_internet_tool = ToolNode([search_internet], name = "search_internet_tool")
summarize_web_content_from_link_tool = ToolNode([summarize_web_content_from_link], name = "summarize_web_content_from_link_tool")

# Step 3: Generate a response using the retrieved content.
def generate_search_response(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
            
    tool_messages = recent_tool_messages[::-1]
    system_message_content = ''
    if tool_messages:
        
        system_message_content = "your google search agent, after searching internet summary in url" + tool_messages[0].content + "in a table, please response in markdown format"
    
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    
    prompt = [SystemMessage(system_message_content)] + conversation_messages if system_message_content else conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}





def generate_visit_link_response(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
            
    tool_messages = recent_tool_messages[::-1]
    system_message_content = ''
    if tool_messages:
        
        system_message_content = "your google search agent, summarize web content" +  tool_messages[0].content + "in a table, please response in markdown format"
    
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    
    prompt = [SystemMessage(system_message_content)] + conversation_messages if system_message_content else conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}




def route_by_message_state(state: MessagesState):
    
    
        message = state["messages"][-1]
        if message.type == "ai" and message.tool_calls:
            return  message.tool_calls[0]["name"]
        return "END"
    
    


# Build graph
graph_builder = StateGraph(MessagesState)

graph_builder.add_node(query_or_respond)
graph_builder.add_node(search_internet_tool)
graph_builder.add_node(summarize_web_content_from_link_tool)
graph_builder.add_node(generate_search_response)
graph_builder.add_node(generate_visit_link_response)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    route_by_message_state,
    {"END": END, "search_internet": "search_internet_tool", "summarize_web_content_from_link":"summarize_web_content_from_link_tool"},
)
graph_builder.add_edge("search_internet_tool", "generate_search_response")
graph_builder.add_edge("summarize_web_content_from_link_tool", "generate_visit_link_response")
graph_builder.add_edge("generate_search_response", END)
graph_builder.add_edge("generate_visit_link_response", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)






