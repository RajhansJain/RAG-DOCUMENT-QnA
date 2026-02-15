RAG-Based Multi-Document QnA System

Python | RAG | FAISS | Gemini API | Streamlit

A Retrieval-Augmented Generation (RAG) based application that allows users to upload multiple PDF documents and perform contextual question-answering using semantic search and Large Language Models.

📌 Overview

This project implements a complete RAG pipeline that enables:

Uploading multiple PDF documents

Extracting and chunking document text

Generating embeddings

Performing semantic search using a vector database

Producing context-aware answers using Gemini LLM

The system retrieves the most relevant document chunks before generating responses, improving factual accuracy and contextual relevance.

🚀 Features

Multi-PDF document upload support

Automated text extraction using PyPDF

Intelligent text chunking for efficient retrieval

Vector embedding generation using Gemini API

FAISS-based semantic search

Context-aware conversational memory

LLM-powered answer generation

Interactive Streamlit user interface

🏗️ Architecture

Document Upload (PDF files)

Text Extraction (PyPDF)

Text Chunking

Embedding Generation (Gemini Embeddings)

Vector Storage (FAISS)

Semantic Retrieval

Context-Aware Response Generation (Gemini LLM)

🛠️ Tech Stack

Python

Streamlit

Gemini API

LangChain

FAISS (Vector Store)

PyPDF

dotenv

📂 Project Structure
RAG-DOCUMENT-QnA/
│
├── DocReader.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/RajhansJain/RAG-DOCUMENT-QnA.git
cd RAG-DOCUMENT-QnA

2️⃣ Create Virtual Environment
python -m venv venv


Activate:

Windows

venv\Scripts\activate


Mac/Linux

source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Configure Environment Variables

Create a .env file in the root directory:

GEMINI_API_KEY=your_api_key_here

5️⃣ Run the Application
streamlit run DocReader.py

🎥 Demo

This application demonstrates:

Multi-document upload

Contextual Q&A over uploaded PDFs

Chat-based interaction

Semantic search powered retrieval

(Demo video can be added here)

🔮 Future Improvements

Cloud deployment (Streamlit Cloud / Render)

Authentication layer

Optimized chunking strategy

Persistent vector storage

UI enhancements

Docker containerization

📬 Contact

Rajhans Jain
B.Tech, Jabalpur Engineering College
Email: rajhansjain19@gmail.com

GitHub: https://github.com/RajhansJain

LinkedIn: https://www.linkedin.com/in/rajhans-jain-790b7a303/
