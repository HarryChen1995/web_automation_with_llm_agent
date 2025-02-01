from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama
from web_automation import google_search, driver, go_to_web_page



llm = ChatOllama(model="llama3.2")





from langchain_core.tools import tool


@tool
def look_up_internet(topic: str):
    """look up internet with a topic."""
    urls = google_search(driver, topic)
    return '\n'.join(urls)


@tool
def summarize_web_content_from_link(link: str):
    """summarize web content from link."""
    pages = go_to_web_page(driver, link)
    return "".join([page.page_content for page in pages])

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([look_up_internet, summarize_web_content_from_link])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
look_up_internet_tool = ToolNode([look_up_internet], name = "look_up_internet_tool")
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
        
        system_message_content = "your google search agent, summarize url results " +  "\n".join([tool_message.content for tool_message in tool_messages]) + "in a table, please response in markdown format"
    
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
        
        system_message_content = "your google search agent, summarize web content" +  "\n".join([tool_message.content for tool_message in tool_messages]) + "in a table, please response in markdown format"
    
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
graph_builder.add_node(look_up_internet_tool)
graph_builder.add_node(summarize_web_content_from_link_tool)
graph_builder.add_node(generate_search_response)
graph_builder.add_node(generate_visit_link_response)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    route_by_message_state,
    {"END": END, "look_up_internet": "look_up_internet_tool", "summarize_web_content_from_link":"summarize_web_content_from_link_tool"},
)
graph_builder.add_edge("look_up_internet_tool", "generate_search_response")
graph_builder.add_edge("summarize_web_content_from_link_tool", "generate_visit_link_response")
graph_builder.add_edge("generate_search_response", END)
graph_builder.add_edge("generate_visit_link_response", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)






