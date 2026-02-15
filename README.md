RAG-Based Multi-Document QnA System

Overview:-

This project implements a Retrieval-Augmented Generation (RAG) pipeline that allows users to upload multiple PDF documents and ask contextual questions. The system uses vector embeddings and semantic search to retrieve relevant document chunks before generating responses using a Large Language Model.


Features:-

Multi-PDF upload
Semantic search using FAISS
Context-aware chat history
LLM-powered answer generation
Streamlit UI interface


Tech Stack:-

Python
Streamlit
Gemini API
LangChain
FAISS
PyPDF


How to Run:-

Clone the repository
Create a virtual environment

Install dependencies:
pip install -r requirements.txt

Create .env file and add:
GEMINI_API_KEY=your_api_key_here

Run:
streamlit run app.py

Future Improvements:-
Cloud deployment
Authentication
Optimized chunking strategy