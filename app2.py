import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from htmlTemplates import bot_template, user_template, css
import os

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorestore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    load_dotenv()
    if os.getenv("GROQ_API_KEY") is None or os.getenv("GROQ_API_KEY") == "":
        print("GROQ_API_KEY is not set")
        exit(1)
    else:
        print("GROQ_API_KEY is set")
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "main_chain" not in st.session_state:
        st.session_state.main_chain = None

    st.header("CHAT with PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.main_chain:
        with st.spinner("Generating answer..."):
            answer = st.session_state.main_chain.invoke(user_question)
        st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", answer), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here...", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorestore(text_chunks)
                st.session_state.vectorstore = vectorstore

                # Build the runnables pipeline
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
                prompt = PromptTemplate(
                    template="""
                    You are a helpful assistant.
                    Use the following context to answer the question.
                    If the context is insufficient, say you don't know.

                    Context: {context}
                    Question: {question}
                    """,
                    input_variables=['context', 'question']
                )
                parallel_chain = RunnableParallel({
                    'context': retriever | RunnableLambda(format_docs),
                    'question': RunnablePassthrough()
                })
                llm = ChatGroq(model_name="llama3-8b-8192")
                parser = StrOutputParser()
                main_chain = parallel_chain | prompt | llm | parser
                st.session_state.main_chain = main_chain

if __name__ == '__main__':
    main() 