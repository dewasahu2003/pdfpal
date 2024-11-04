import streamlit as st
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

def create_mixtral_chatbot(model_path="mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"):
    """
    Creates a local chatbot using Mixtral model
    Memory Usage: ~10GB total
    Args:
        model_path: Path to the quantized Mixtral model file
    Returns:
        conversation: ConversationChain object
    """
    # Callbacks for streaming output
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    # Initialize the Mixtral model
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.7,
        max_tokens=512,
        n_ctx=2048,
        n_batch=512,
        callback_manager=callback_manager,
        n_threads=8,  # Adjust based on your CPU
        n_gpu_layers=0,  # CPU only
        verbose=True,
        grammar_path=None
    )
    
    # Use window memory to manage memory usage
    memory = ConversationBufferWindowMemory(k=5)  # Keep last 5 exchanges
    
    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    return conversation

# Initialize the chatbot
chatbot = create_mixtral_chatbot()

# Streamlit app layout
st.title("Mixtral Chatbot")
st.write("Talk to the chatbot! Type your message below.")

# Input box for user message
user_input = st.text_input("You:", "")

# Handle user input and display response
if user_input:
    try:
        response = chatbot.predict(input=user_input)
        st.write("Bot:", response)
    except Exception as e:
        st.error(f"Error: {e}")
        st.write("Retrying with a shorter response...")
        response = chatbot.predict(input=user_input[:256])  # Retry with truncated input
        st.write("Bot:", response)

