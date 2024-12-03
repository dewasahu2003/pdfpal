# PdfPaL : Local AI Document Interaction

## 📹 Demo Video
[Watch the Full Usage Video](https://github.com/user-attachments/assets/1842d6b8-dc17-4801-88a8-3a7f6fdf4775)

## 🌟 Features
- Chat with multiple PDF documents simultaneously
- Local AI processing (no cloud dependencies)
- Easy-to-use Streamlit interface
- Supports various document types
- Quick setup with Docker and Python

## 🛠️ Prerequisites
- Docker
- Python 3.8+
- Git
- Minimum 8GB RAM 

## 🚀 Quick Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/pdf-chat.git
cd pdf-chat
```

### 2. Start Docker
Launch the required containers:
```bash
docker compose up
```

### 3. Set Up Python Environment
Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r reqtiny.txt
```

### 5. Download AI Model
Run the model installation script:
```bash
# On Windows
./tiny-windows.bat

```

### 6. Launch Application
```bash
streamlit run tiny_rag.py
```

## 💻 Usage
1. Open your browser and navigate to `http://localhost:8501`
2. Upload PDF documents
3. Start chatting with your documents!

## 🔧 Troubleshooting
- Ensure Docker is running
- Check Python version compatibility
- Verify all dependencies are installed
- Restart the application if encountering issues

## ⚠️ System Requirements
- CPU: Multi-core processor
- RAM: 8GB+ 
- Storage: 10GB free space
- Operating System: Windows 10/11, macOS, Linux



## 🤝 Contributing
Contributions are welcome! Please read our contributing guidelines before getting started.

## 🙏 Acknowledgements
- Streamlit
- Hugging Face
- Docker
- Python Community
