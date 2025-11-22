import streamlit as st
import time
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

import os
from dotenv import load_dotenv

load_dotenv()
llama_api_key = os.getenv("LLAMA_API_KEY")

if not llama_api_key:
    raise ValueError("API key is missing! Check your .env file.")


# Initialize Streamlit app
st.title("Chat with Llama 2 on Your Documents")

# Initialize Llama model
llm = ChatGroq(
    groq_api_key=os.getenv("LLAMA_API_KEY"),
    model_name="llama2-70b-chat"  # Change model if needed
)

# Define prompt template
prompt = ChatPromptTemplate.from_template("""
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.

    <context>
    {context}
    </context>

    Question: {input}
""")

def vector_embeddings():
    """Create vector embeddings from PDF documents using HuggingFace embeddings."""
    try:
        if "vectors" not in st.session_state:
            with st.spinner("Creating vector embeddings..."):
                # Initialize embeddings (HuggingFace instead of OpenAI)
                st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

                # Load documents
                st.session_state.loader = PyPDFDirectoryLoader("./STUDY")
                st.session_state.docs = st.session_state.loader.load()

                # Create text splitter
                st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                # Split documents
                st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                    st.session_state.docs[:20]
                )

                # Create vector store
                st.session_state.vectors = FAISS.from_documents(
                    st.session_state.final_documents,
                    st.session_state.embeddings
                )

            st.success("Vector store DB is ready!")

    except Exception as e:
        st.error(f"An error occurred while creating embeddings: {str(e)}")

# Create input field for questions
prompt1 = st.text_input("Enter your questions from the Documents")

# Create button for document embedding
if st.button("Create Document Embeddings"):
    vector_embeddings()

# Process the query
if prompt1:
    try:
        if "vectors" not in st.session_state:
            st.warning("Please create document embeddings first!")
        else:
            start = time.process_time()

            # Create document chain
            document_chain = create_stuff_documents_chain(llm, prompt)

            # Create retriever
            retriever = st.session_state.vectors.as_retriever()

            # Create and invoke retrieval chain
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            response = retrieval_chain.invoke({'input': prompt1})

            # Display processing time
            st.info(f"Response time: {time.process_time() - start:.2f} seconds")

            # Display response
            st.write(response['answer'])

            # Display similar documents
            with st.expander("Documents Similarity Search"):
                for i, doc in enumerate(response['context']):
                    st.write(f"Document {i + 1}:")
                    st.write(doc.page_content)
                    st.write("----------------------------")

    except Exception as e:
        st.error(f"An error occurred while processing your question: {str(e)}")
