import streamlit as st
from langchain.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
import os

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
                'max_new_tokens': 128,
                'temperature': 0.7,
                'context_length': 512,
                'threads': 2,
                'batch_size': 8,
                'stream': True,
                'gpu_layers': 0
            }
        )
        
        # Load and process documents
        self.load_documents(docs_dir)

    def load_documents(self, docs_dir):
        """Load and process PDF documents in small batches."""
        documents = []
        
        # Load all PDF files from the directory
        for filename in os.listdir(docs_dir):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(docs_dir, filename)
                st.write(f"loading {filename}")
                st.write(f"loading {pdf_path}")
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
        
        # Initialize the vector store
        self.vectorstore = Qdrant.from_documents(
            documents=texts,
            embedding=self.embeddings,
            url="http://localhost:6333",
            collection_name="knowledge_base",
            force_recreate=True
        )
        
        # Use minimal window memory
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=2,
            return_messages=True
        )
        
        # Create retrieval chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 2}),
            memory=memory,
            verbose=True
        )

    def chat(self, query: str) -> str:
        """Process a query and return a response."""
        try:
            response = self.chain({"question": query})
            return response["answer"]
        except Exception as e:
            print(f"Error: {e}")
            # Retry with shorter input
            response = self.chain({"question": query[:256]})
            return response["answer"]

def main():
    """Streamlit app for the TinyLlama RAG Chatbot."""
    st.title("TinyLlama RAG Chatbot")
    st.write("Talk to the chatbot! Type your message below.")
    
    # Initialize RAG chatbot
    docs_dir = "./knowledge_base"
    chatbot = TinyRAGChatbot(docs_dir)
    
    # Input box for user message
    user_input = st.text_input("You:", "")
    
    # Handle user input and display response
    if user_input:
        response = chatbot.chat(user_input)
        st.write("Bot:", response)

if __name__ == "__main__":
    main()
