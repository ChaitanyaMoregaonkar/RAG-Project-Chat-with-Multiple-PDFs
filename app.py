from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import login
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEndpoint


def get_pdf_text(uploaded_files):
    # Placeholder for actual PDF text extraction logic
    text=""
    
    for pdf in uploaded_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text  


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store


def get_conversation_chain(vector_store):
    
    llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature=0.01,
    max_new_tokens=512  
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    conversation_chain = ConversationalRetrievalChain.from_llm( 
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory,
        return_source_documents=True,
    )

    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"**User:** {message.content}")
        else:
            st.write(f"**Assistant:** {message.content}")


def main():
    load_dotenv()
    login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:", layout="wide")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs")
    user_question = st.text_input("Enter your question here:")
    if user_question:
        handle_userinput(user_question) 
    

    with st.sidebar:
        st.subheader("Upload PDF files")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        if st.button("Process PDFs"):
            with st.spinner("Processing..."):
                #get pdf text
                raw_text = get_pdf_text(uploaded_files)  
                
                #get text chunks
                text_chunks = get_text_chunks(raw_text) 
                
                #create vector store
                vector_store = get_vector_store(text_chunks)   
                
                #create conversation chain
                st.session_state.conversation= get_conversation_chain(vector_store)  
    

if __name__ == "__main__":
    main()
