import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms import LlamaCpp
import qdrant_client
import os

class MixtralRAGChatbot:
    def __init__(self, docs_dir, model_path="mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"):
        """
        Initialize RAG chatbot with Mixtral.
        Memory Usage: ~12GB total (excluding documents).
        Args:
            docs_dir: Directory containing documents.
            model_path: Path to the quantized Mixtral model file.
        """
        # Use BAAI embedding model for better performance
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize Qdrant client
        self.qdrant = QdrantClient(url="http://localhost:6333", prefer_grpc=False)

        # Initialize Mixtral model
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        # Initialize the Mixtral model
        self.llm = LlamaCpp(
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
        
        # Load and process documents
        self.load_documents(docs_dir)

    def load_documents(self, docs_dir):
        """Load and process PDF documents in batches."""
        documents = []
        
        # Load all PDF files from the directory
        for filename in os.listdir(docs_dir):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(docs_dir, filename)
                st.write(f"Loading {filename}...")
                loader = UnstructuredPDFLoader(pdf_path)
                docs = loader.load()
                if not docs:
                    raise ValueError(f"No documents were loaded from the PDF: {filename}")
                documents.extend(docs)

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=250,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)
        
        # Create vector store with disk persistence
        self.vectorstore = Qdrant.from_documents(
            documents=texts,
            embedding=self.embeddings,
             url="http://localhost:6333",
            collection_name="knowledge_base",
            force_recreate=True
        )
        
        # Use window memory
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,
            return_messages=True
        )
        
        # Create retrieval chain with Mixtral
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 4}  # Number of relevant chunks to retrieve
            ),
            memory=memory,
            verbose=True,
            return_source_documents=True  # Include source information
        )
    
    def chat(self, query: str) -> dict:
        """Process a query and return response with sources."""
        try:
            result = self.chain({"question": query})
            return {
                'answer': result['answer'],
                'sources': [doc.metadata.get('source', 'Unknown') 
                          for doc in result.get('source_documents', [])]
            }
        except Exception as e:
            print(f"Error: {e}")
            # Retry with more conservative settings
            self.llm.max_tokens = 256
            result = self.chain({"question": query})
            return {
                'answer': result['answer'],
                'sources': [doc.metadata.get('source', 'Unknown') 
                          for doc in result.get('source_documents', [])]
            }

# Streamlit app layout
def main():
    st.title("Mixtral RAG Chatbot")
    st.write("Talk to the chatbot! Type your message below.")

    # Initialize RAG chatbot
    docs_dir = "./knowledge_base"
    chatbot = MixtralRAGChatbot(docs_dir)

    # Input box for user message
    user_input = st.text_input("You:", "")

    # Handle user input and display response
    if user_input:
        result = chatbot.chat(user_input)
        st.write("**Bot:**", result['answer'])
        if result['sources']:
            st.write("**Sources:**", ', '.join(result['sources']))

if __name__ == "__main__":
    main()
