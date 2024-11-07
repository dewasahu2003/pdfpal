import streamlit as st
from langchain.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
import os
import time

st.set_page_config(layout="wide") 
class TinyRAGChatbot:
    def __init__(self, docs_dir, model_path="tinyllama-1.1b-chat-v1.0.gguf"):
        """
        Initialize lightweight RAG chatbot with TinyLlama.
        Memory Usage: ~1.5GB total (excluding documents).
        """
        # Use tiny embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="paraphrase-MiniLM-L3-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize Qdrant client
        self.qdrant = QdrantClient(url="http://localhost:6333", prefer_grpc=False)

        # Initialize TinyLlama with ctransformers
        self.llm = CTransformers(
            model=model_path,
            model_type="llama",
            config={
                'max_new_tokens': 256,
                'temperature': 0.5,
                'context_length': 2048,  # Increase the context length
                'threads': 8,
                'batch_size': 8,
                'stream': False,
                'gpu_layers': 0,
                'seed': 42,
            }
        )
        self.processed_files = []
        
        # Load and process documents
        self.load_documents(docs_dir)
    
    def update_vectordb(self,pdf_path,filename):
        self.processed_files.append(filename)
        with st.spinner("Updating the vector db.... including new file"):
            try:
                loader = UnstructuredPDFLoader(pdf_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=250,
                    length_function=len
                )
                texts = text_splitter.split_documents(docs)
                self.vectorstore.add_documents(texts)
               # st.success("updated the vectordb")
                
            except Exception as e:
                st.error(f'An error occurred: {str(e)}')
                
                
    def load_documents(self, docs_dir):
        """Load and process PDF documents in small batches."""
        documents = []
        
        # Check if the docs_dir is empty
        if not os.listdir(docs_dir):
            st.write("No documents found in the knowledge base directory.")
            return
            # Count total PDF files
        pdf_files = [f for f in os.listdir(docs_dir) if f.endswith('.pdf')]
        total_files = len(pdf_files)
            
        # Create a spinner that shows while documents are being processed
        with st.spinner('Processing documents... Please wait.'):
            progress_bar = st.progress(0)
            try:
                # Load all PDF files from the directory
                for idx,filename in enumerate(pdf_files):
                    pdf_path = os.path.join(docs_dir, filename)
                   # st.write(f"Loading {filename}")
                    loader = UnstructuredPDFLoader(pdf_path)
                    docs = loader.load()
                    if not docs:
                        raise ValueError(f"No documents were loaded from the PDF: {filename}")
                    documents.extend(docs)
                    progress = (idx + 1) / total_files
                    progress_bar.progress(progress)
                    
                # Show progress bar for text splitting
                #st.write("Splitting documents into chunks...")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=250,
                    length_function=len
                )
                texts = text_splitter.split_documents(documents)
                progress_bar.progress(0.8)
                # Show progress for vector store initialization
              #  st.write("Initializing vector store...")
                self.vectorstore = Qdrant.from_documents(
                    documents=texts,
                    embedding=self.embeddings,
                    url="http://localhost:6333",
                    collection_name="knowledge_base",
                    force_recreate=True
                )
                
                # Initialize memory and chain
                memory = ConversationBufferWindowMemory(
                    memory_key="chat_history",
                    k=2,
                    return_messages=True
                )
                progress_bar.progress(0.9)
                self.chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.vectorstore.as_retriever(search_kwargs={"k": 2}),
                    memory=memory,
                    verbose=True
                )
                progress_bar.progress(1.0)
                # Show success message
                
                
            except Exception as e:
                st.error(f'An error occurred: {str(e)}')
            finally:
            # Clean up the progress bar
               
                progress_bar.empty()
                
                
    def chat(self, query: str) -> str:
        try:
            response = self.chain.invoke({"question": query})

            
            answer = response["answer"]

            # Display the reference information as small boxes
            # with st.container():
            #     for result in response["result"]:
            #         with st.expander(result.get("source_text", "Reference")):
            #             st.write(result.get("page_content", ""))

            return answer
        except Exception as e:
            st.write(f"Error: {e}")
            print(f"Error: {e}")
            # Retry with shorter input
            response = self.chain({"question": query[:256]})
            return response["answer"]
        
docs_dir = "./knowledge_base"
chatbot = TinyRAGChatbot(docs_dir)

def main():
    """Streamlit app for the TinyLlama RAG Chatbot."""
     # Set the page layout to wide
    
    st.title("TinyLlama RAG Chatbot")

    # Initialize RAG chatbot
    # File upload component
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Display the conversation history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.conversation_history:
            if message.startswith("You:"):
                st.write(f"<div style='text-align: right;'>{message}</div>", unsafe_allow_html=True)
                st.write("")
                
            else:
                st.write(f"<div>{message}</div>", unsafe_allow_html=True)
                st.write("")

    # Input box for user message
    user_input = st.text_input("You:", "", key="user_input")

    # Handle user input and display response
    if user_input:
        response = chatbot.chat(user_input)
        st.session_state.conversation_history.append(f"You: {user_input}")
        st.session_state.conversation_history.append(f"Bot: {response}")
        chat_container.empty()
        with chat_container:
            for message in st.session_state.conversation_history:
                if message.startswith("You:"):
                    st.write(f"<div style='text-align: right;'>{message}</div>", unsafe_allow_html=True)
                    st.write("")
                    
                else:
                    st.write(f"<div>{message}</div>", unsafe_allow_html=True)
                    st.write("")

    # Process uploaded files and update the knowledge base
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in chatbot.processed_files:
                pdf_path = os.path.join(docs_dir, file.name)
                with open(pdf_path, "wb") as f:
                    f.write(file.getbuffer())
                #st.write(f"Uploaded and processed {file.name}")
                chatbot.update_vectordb(pdf_path,file.name)
       
        
       
if __name__ == "__main__":
    main()
