from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Google Gemini API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize LangChain's Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

def generate(state: MessagesState):
    """Generate answer with system message context."""
    system_message = SystemMessage(
        content=(
            "You are an assistant for question-answering tasks. "
            "Answer concisely in 3 sentences maximum. "
            "If you don't know the answer, say you don't know."
        )
    )
    
    # Convert any dictionary messages to proper Message objects
    messages = []
    for msg in state["messages"]:
        if isinstance(msg, dict):
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        else:
            messages.append(msg)
    
    # Combine system message with conversation history
    prompt = [system_message] + messages
    
    # Generate response
    response = llm.invoke(prompt)
    return {"messages": [response]}

# Build graph
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("generate", generate)
graph_builder.set_entry_point("generate")
graph_builder.add_edge("generate", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# Conversation thread configuration
config = {"configurable": {"thread_id": "abc123"}}

while True:
    try:
        user_input = input("\nAsk Gemini a question (q to quit): ")
        if user_input.lower() == 'q':
            break
            
        # Create proper HumanMessage object
        user_message = HumanMessage(content=user_input)
        
        # Stream the response
        for step in graph.stream(
            {"messages": [user_message]},
            config=config,
            stream_mode="values",
        ):
            if isinstance(step["messages"][-1], AIMessage):
                print(f"\nAssistant: {step["messages"][-1].content}")
                    
    except KeyboardInterrupt:
        break

print("Goodbye!")