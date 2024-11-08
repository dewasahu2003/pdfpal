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
        self.embeddings = HuggingFaceEmbeddings(
            model_name="paraphrase-MiniLM-L3-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.qdrant = QdrantClient(url="http://localhost:6333", prefer_grpc=False)
        self.llm = CTransformers(
            model=model_path,
            model_type="llama",
            config={
                'max_new_tokens': 256,
                'temperature': 0.5,
                'context_length': 2048,
                'threads': 8,
                'batch_size': 8,
                'stream': False,
                'gpu_layers': 0,
                'seed': 42,
            }
        )
        self.processed_files = []
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
            except Exception as e:
                st.error(f'An error occurred: {str(e)}')
                
    def load_documents(self, docs_dir):
        documents = []
        if not os.listdir(docs_dir):
            st.write("No documents found in the knowledge base directory.")
            return
        pdf_files = [f for f in os.listdir(docs_dir) if f.endswith('.pdf')]
        total_files = len(pdf_files)
        with st.spinner('Processing documents... Please wait.'):
            progress_bar = st.progress(0)
            try:
                for idx,filename in enumerate(pdf_files):
                    pdf_path = os.path.join(docs_dir, filename)
                    loader = UnstructuredPDFLoader(pdf_path)
                    docs = loader.load()
                    if not docs:
                        raise ValueError(f"No documents were loaded from the PDF: {filename}")
                    documents.extend(docs)
                    progress = (idx + 1) / total_files
                    progress_bar.progress(progress)
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=250,
                    length_function=len
                )
                texts = text_splitter.split_documents(documents)
                progress_bar.progress(0.8)
                self.vectorstore = Qdrant.from_documents(
                    documents=texts,
                    embedding=self.embeddings,
                    url="http://localhost:6333",
                    collection_name="knowledge_base",
                    force_recreate=True
                )
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
            except Exception as e:
                st.error(f'An error occurred: {str(e)}')
            finally:
                progress_bar.empty()
                
    def chat(self, query: str) -> str:
        try:
            response = self.chain.invoke({"question": query})
            answer = response["answer"]
            return answer
        except Exception as e:
            st.write(f"Error: {e}")
            print(f"Error: {e}")
            response = self.chain({"question": query[:256]})
            return response["answer"]
        
docs_dir = "./knowledge_base"
chatbot = TinyRAGChatbot(docs_dir)

def main():
    st.title("TinyLlama RAG Chatbot")
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.conversation_history:
            if message.startswith("You:"):
                st.write(f"<div style='text-align: right;'>{message}</div>", unsafe_allow_html=True)
                st.write("")
            else:
                st.write(f"<div>{message}</div>", unsafe_allow_html=True)
                st.write("")

    user_input = st.text_input("You:", "", key="user_input")

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

    if uploaded_files:
        for file in uploaded_files:
            if file.name not in chatbot.processed_files:
                pdf_path = os.path.join(docs_dir, file.name)
                with open(pdf_path, "wb") as f:
                    f.write(file.getbuffer())
                chatbot.update_vectordb(pdf_path,file.name)
       
if __name__ == "__main__":
    main()
