import streamlit as st
from langchain_community.llms.ctransformers import CTransformers
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

def create_tiny_chatbot(model_path="tinyllama-1.1b-chat-v1.0.gguf"):
    """
    Creates a lightweight local chatbot using TinyLlama with ctransformers.
    Memory Usage: ~800MB total.
    """
    llm = CTransformers(
        model=model_path,
        model_type="llama",
        config={
            'max_new_tokens': 128,
            'temperature': 0.7,
            'context_length': 512,
            'threads': 2,
            'batch_size': 8,
            'stream': True,
            'gpu_layers': 0
        }
    )
    
    memory = ConversationBufferWindowMemory(k=2)
    
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )
    
    return conversation

# Initialize the chatbot
chatbot = create_tiny_chatbot()

# Streamlit app layout
st.title("TinyLlama Chatbot")
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
        st.write("Retrying with shorter response...")
        response = chatbot.predict(input=user_input[:256])
        st.write("Bot:", response)

