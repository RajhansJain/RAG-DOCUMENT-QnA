# streamlit run DocReader.py

import os
import hashlib
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

# --------------------------------------------------
# 1. Load API Key
# --------------------------------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("❌ GOOGLE_API_KEY not found")
    st.stop()

client = genai.Client(api_key=API_KEY)

# --------------------------------------------------
# 2. Utility
# --------------------------------------------------
def file_id(file):
    """Unique hash for each uploaded PDF"""
    return hashlib.md5(file.getvalue()).hexdigest()

# --------------------------------------------------
# 3. PDF Processing
# --------------------------------------------------
def extract_pdf_chunks(uploaded_file, chunk_size=500):
    reader = PdfReader(uploaded_file)
    chunks = []

    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        words = text.split()

        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i + chunk_size]).strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": page_no,
                    "source": uploaded_file.name
                })

    return chunks

# --------------------------------------------------
# 4. Embeddings (FREE)
# --------------------------------------------------
def embed_texts(texts):
    if not texts:
        return np.array([])

    response = client.models.embed_content(
        # model="models/text-embedding-004",
        model="models/gemini-embedding-001",
        contents=texts
    )

    return np.array([e.values for e in response.embeddings])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --------------------------------------------------
# 5. Retrieve Relevant Chunks
# --------------------------------------------------
def retrieve_chunks(query, chunks, embeddings, top_k=3):
    query_embedding = embed_texts([query])

    if query_embedding.size == 0:
        return []

    query_embedding = query_embedding[0]

    scores = [
        cosine_similarity(query_embedding, emb)
        for emb in embeddings
    ]

    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# --------------------------------------------------
# 6. Gemini Q&A
# --------------------------------------------------
def ask_gemini(context_chunks, question):
    if not context_chunks:
        return "❌ No relevant context found in the uploaded PDFs."

    context_text = "\n\n".join(
        [f"(PDF: {c['source']}, Page {c['page']}) {c['text']}"
         for c in context_chunks]
    )

    prompt = f"""
You are a PDF document assistant.
Answer ONLY using the provided context.
Mention PDF name and page numbers when relevant.

CONTEXT:
{context_text}

QUESTION:
{question}
"""

    response = client.models.generate_content(
        model="models/gemini-flash-latest",
        contents=prompt
    )

    return response.text

# --------------------------------------------------
# 7. Streamlit UI
# --------------------------------------------------
st.set_page_config(
    page_title="Gemini PDF Chat Reader",
    layout="wide"
)

st.title("📘 Gemini PDF Chat Reader")

with st.sidebar:
    st.header("📄 Upload PDFs")
    uploaded_pdfs = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

# --------------------------------------------------
# 8. Session State Init
# --------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = []

if "pdf_embeddings" not in st.session_state:
    st.session_state.pdf_embeddings = np.empty((0, 3072))

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# --------------------------------------------------
# 9. Process ONLY New PDFs
# --------------------------------------------------
if uploaded_pdfs:
    new_files = []

    for pdf in uploaded_pdfs:
        fid = file_id(pdf)
        if fid not in st.session_state.processed_files:
            new_files.append((pdf, fid))

    if new_files:
        with st.spinner("📥 Processing new PDFs..."):
            for pdf, fid in new_files:
                chunks = extract_pdf_chunks(pdf)

                if not chunks:
                    st.warning(f"⚠️ No readable text in {pdf.name}")
                    continue

                texts = [c["text"] for c in chunks]
                embeddings = embed_texts(texts)

                if embeddings.size == 0:
                    st.warning(f"⚠️ Embedding failed for {pdf.name}")
                    continue

                st.session_state.pdf_chunks.extend(chunks)
                st.session_state.pdf_embeddings = np.vstack(
                    [st.session_state.pdf_embeddings, embeddings]
                )
                st.session_state.processed_files.add(fid)

        st.success(f"✅ Added {len(new_files)} new PDF(s)")

# --------------------------------------------------
# 10. Chat Interface
# --------------------------------------------------
if st.session_state.pdf_chunks:

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    user_question = st.chat_input("Ask a question from the PDFs")

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.spinner("🔎 Searching documents..."):
            relevant_chunks = retrieve_chunks(
                user_question,
                st.session_state.pdf_chunks,
                st.session_state.pdf_embeddings
            )
            answer = ask_gemini(relevant_chunks, user_question)

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.chat_history.append(("user", user_question))
        st.session_state.chat_history.append(("assistant", answer))

else:
    st.info("👈 Upload PDFs to start chatting")


# ----------------------------------------------------------------------------------------------
# the above code is the upgraded version of the below code
# ----------------------------------------------------------------------------------------------
# # streamlit run DocReader.py

# import os
# import numpy as np
# import streamlit as st
# from dotenv import load_dotenv
# from google import genai
# from pypdf import PdfReader

# # --------------------------------------------------
# # 1. Load API Key
# # --------------------------------------------------
# load_dotenv()
# API_KEY = os.getenv("GOOGLE_API_KEY")

# if not API_KEY:
#     st.error("GOOGLE_API_KEY not found")
#     st.stop()

# client = genai.Client(api_key=API_KEY)

# # --------------------------------------------------
# # 2. PDF Processing
# # --------------------------------------------------
# def extract_pdf_chunks(uploaded_file, chunk_size=500):
#     reader = PdfReader(uploaded_file)
#     chunks = []

#     for page_no, page in enumerate(reader.pages, start=1):
#         text = page.extract_text() or ""
#         words = text.split()

#         for i in range(0, len(words), chunk_size):
#             chunk_text = " ".join(words[i:i + chunk_size])
#             chunks.append({
#                 "text": chunk_text,
#                 "page": page_no
#             })

#     return chunks

# # --------------------------------------------------
# # 3. Embeddings (FREE)
# # --------------------------------------------------
# def embed_texts(texts):
#     response = client.models.embed_content(
#         model="models/text-embedding-004",
#         contents=texts
#     )
#     return np.array([e.values for e in response.embeddings])

# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# # --------------------------------------------------
# # 4. Retrieve Relevant Chunks
# # --------------------------------------------------
# def retrieve_chunks(query, chunks, embeddings, top_k=3):
#     query_embedding = embed_texts([query])[0]

#     scores = [
#         cosine_similarity(query_embedding, emb)
#         for emb in embeddings
#     ]

#     top_indices = np.argsort(scores)[-top_k:][::-1]

#     return [chunks[i] for i in top_indices]

# # --------------------------------------------------
# # 5. Gemini Q&A
# # --------------------------------------------------
# def ask_gemini(context_chunks, question):
#     context_text = "\n\n".join(
#         [f"(Page {c['page']}) {c['text']}" for c in context_chunks]
#     )

#     prompt = f"""
# You are a PDF document assistant.
# Answer ONLY using the provided context.
# Mention page numbers when relevant.

# CONTEXT:
# {context_text}

# QUESTION:
# {question}
# """

#     response = client.models.generate_content(
#         model="models/gemini-flash-latest",
#         contents=prompt
#     )

#     return response.text

# # --------------------------------------------------
# # 6. Streamlit UI
# # --------------------------------------------------
# st.set_page_config(
#     page_title="Gemini PDF Chat Reader",
#     layout="wide"
# )

# st.title("📘 Gemini PDF Chat Reader (2026)")

# with st.sidebar:
#     st.header("📄 Upload PDF")
#     uploaded_pdf = st.file_uploader(
#         "Choose a PDF file",
#         type=["pdf"]
#     )

# # Session state
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# if "pdf_chunks" not in st.session_state:
#     st.session_state.pdf_chunks = None

# if "pdf_embeddings" not in st.session_state:
#     st.session_state.pdf_embeddings = None

# # --------------------------------------------------
# # 7. Load PDF
# # --------------------------------------------------
# if uploaded_pdf and st.session_state.pdf_chunks is None:
#     with st.spinner("Processing PDF..."):
#         chunks = extract_pdf_chunks(uploaded_pdf)
#         texts = [c["text"] for c in chunks]
#         embeddings = embed_texts(texts)

#         st.session_state.pdf_chunks = chunks
#         st.session_state.pdf_embeddings = embeddings

#     st.success("PDF processed and indexed successfully!")

# # --------------------------------------------------
# # 8. Chat Interface
# # --------------------------------------------------
# if st.session_state.pdf_chunks:

#     for role, msg in st.session_state.chat_history:
#         with st.chat_message(role):
#             st.markdown(msg)

#     user_question = st.chat_input("Ask a question from the PDF")

#     if user_question:
#         with st.chat_message("user"):
#             st.markdown(user_question)

#         with st.spinner("Searching document..."):
#             relevant_chunks = retrieve_chunks(
#                 user_question,
#                 st.session_state.pdf_chunks,
#                 st.session_state.pdf_embeddings
#             )

#             answer = ask_gemini(relevant_chunks, user_question)

#         with st.chat_message("assistant"):
#             st.markdown(answer)

#         st.session_state.chat_history.append(("user", user_question))
#         st.session_state.chat_history.append(("assistant", answer))
# else:
#     st.info("Upload a PDF to start chatting 📄")
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------

















# ---------------------------------------------------
# Previous version of the code for reference
# before 2024
# ---------------------------------------------------
# import os
# import streamlit as st
# from pypdf import PdfReader
# from dotenv import load_dotenv
# import google.generativeai as genai
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # ---------------- ENV ----------------
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # ---------------- PDF TEXT ----------------
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         reader = PdfReader(pdf)
#         for page in reader.pages:
#             if page.extract_text():
#                 text += page.extract_text()
#     return text

# # ---------------- CHUNKING ----------------
# def get_text_chunks(text):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1500,
#         chunk_overlap=200
#     )
#     return splitter.split_text(text)

# # ---------------- QUESTION ANSWERING ----------------
# def answer_question(question, text_chunks):
#     # limit chunks to avoid token explosion
#     context = "\n\n".join(text_chunks[:5])

#     prompt = f"""
# You are a helpful AI assistant.
# Answer the question ONLY using the context below.
# If the answer is not present, say:
# "Answer is not available in the provided context."

# Context:
# {context}

# Question:
# {question}

# Answer:
# """

#     model = genai.GenerativeModel("models/gemini-flash-latest")
#     response = model.generate_content(prompt)

#     return response.text

# # ---------------- STREAMLIT UI ----------------
# def main():
#     st.set_page_config(
#         page_title="PDF Document Reader (Gemini)",
#         layout="centered"
#     )

#     st.header("📄 PDF Document Reader using Gemini (Free Tier)")

#     if "text_chunks" not in st.session_state:
#         st.session_state.text_chunks = None

#     user_question = st.text_input("Ask a question from the PDF")

#     if user_question and st.session_state.text_chunks:
#         with st.spinner("Generating answer..."):
#             answer = answer_question(
#                 user_question,
#                 st.session_state.text_chunks
#             )
#             st.subheader("📌 Answer")
#             st.write(answer)

#     with st.sidebar:
#         st.title("📂 Upload PDFs")
#         pdf_docs = st.file_uploader(
#             "Upload PDF files",
#             type=["pdf"],
#             accept_multiple_files=True
#         )

#         if st.button("Submit & Process"):
#             if not pdf_docs:
#                 st.warning("Please upload at least one PDF.")
#             else:
#                 with st.spinner("Reading PDF..."):
#                     raw_text = get_pdf_text(pdf_docs)
#                     chunks = get_text_chunks(raw_text)
#                     st.session_state.text_chunks = chunks
#                     st.success("PDF processed successfully!")

# # ---------------- RUN ----------------
# if __name__ == "__main__":
#     main()
































# ☠️❌🚫-------☠️❌🚫-------☠️❌🚫------- ☠️❌🚫------- ☠️❌🚫------- ☠️❌🚫------- ☠️❌🚫
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# the below 2 codes are out dated and most of ther langchain imports are changed,
# so please ignore the below code and refer to the above code for the latest implementation 
# of the PDF document reader using Gemini Pro and Streamlit UI. 
# The above code is tested and working fine with the latest versions of the libraries.
# ----------------------------------------------------------------------------------------------
# # streamlit run DocReader.py
# import os
# import streamlit as st
# from PyPDF2 import PdfReader
# from dotenv import load_dotenv
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain.prompts import PromptTemplate
# from langchain.chains.question_answering import load_qa_chain
# from langchain.text_spliter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI

# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# def get_pdf_text(pdf_doc):
#     text=''
#     pdf_reader= PdfReader(pdf_doc)
#     for page in pdf_reader.pages:
#         text+= page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="models/gemini-flash-latest",
#                              temperature=0.3)

#     prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain

# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
#     new_db = FAISS.load_local("faiss_index", embeddings)
#     docs = new_db.similarity_search(user_question)

#     chain = get_conversational_chain()

    
#     response = chain(
#         {"input_documents":docs, "question": user_question}
#         , return_only_outputs=True)

#     print(response)
#     st.write("Reply: ", response["output_text"])

# def main():
#     st.set_page_config("Chat PDF")
#     st.header("Chat with PDF using Gemini💁")

#     user_question = st.text_input("Ask a Question from the PDF Files")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu:")
#         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Done")



# if __name__ == "__main__":
#     main()

# ----------------------------------------------------------------------------------------------

# streamlit run DocReader.py

# import os
# import streamlit as st
# from pypdf import PdfReader
# from dotenv import load_dotenv

# import google.generativeai as genai

# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import (
#     GoogleGenerativeAIEmbeddings,
#     ChatGoogleGenerativeAI
# )

# # ---------------- ENV ----------------
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # ---------------- PDF TEXT ----------------
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         reader = PdfReader(pdf)
#         for page in reader.pages:
#             if page.extract_text():
#                 text += page.extract_text()
#     return text

# # ---------------- CHUNKING ----------------
# def get_text_chunks(text):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     )
#     return splitter.split_text(text)

# # ---------------- VECTOR STORE ----------------
# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/embedding-001"
#     )
#     db = FAISS.from_texts(text_chunks, embedding=embeddings)
#     db.save_local("faiss_index")

# # ---------------- USER QUESTION ----------------
# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/embedding-001"
#     )

#     db = FAISS.load_local(
#         "faiss_index",
#         embeddings,
#         allow_dangerous_deserialization=True
#     )

#     docs = db.similarity_search(user_question, k=4)

#     context = "\n\n".join([doc.page_content for doc in docs])

#     prompt = f"""
# You are a helpful AI assistant.
# Answer the question strictly using the context below.
# If the answer is not present, say:
# "Answer is not available in the provided context."

# Context:
# {context}

# Question:
# {user_question}

# Answer:
# """

#     llm = ChatGoogleGenerativeAI(
#         model="models/gemini-flash-latest",
#         temperature=0.3
#     )

#     response = llm.invoke(prompt)

#     st.subheader("📌 Answer")
#     st.write(response.content)

# # ---------------- STREAMLIT UI ----------------
# def main():
#     st.set_page_config(
#         page_title="Chat with PDF (Gemini)",
#         layout="centered"
#     )

#     st.header("📄 Chat with PDF using Gemini (Free Model)")

#     user_question = st.text_input("Ask a question from the PDF")

#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("📂 Upload PDFs")
#         pdf_docs = st.file_uploader(
#             "Upload PDF files",
#             type=["pdf"],
#             accept_multiple_files=True
#         )

#         if st.button("Submit & Process"):
#             if not pdf_docs:
#                 st.warning("Please upload at least one PDF.")
#             else:
#                 with st.spinner("Processing..."):
#                     raw_text = get_pdf_text(pdf_docs)
#                     chunks = get_text_chunks(raw_text)
#                     get_vector_store(chunks)
#                     st.success("PDF processed successfully!")

# # ---------------- RUN ----------------
# if __name__ == "__main__":
#     main()

