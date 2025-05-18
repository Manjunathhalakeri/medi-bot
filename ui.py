import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import streamlit as st

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = OpenAIEmbeddings()
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def rebuild_vectorstore(data_path="data", db_path=DB_FAISS_PATH):
    loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    embedding_model = OpenAIEmbeddings()
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(db_path)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm():
    llm = ChatOpenAI(
        temperature=0.5,
        model_name="gpt-3.5-turbo",  # or "gpt-4" if you have access
        max_tokens=512
    )
    return llm

def save_uploaded_file(uploaded_file, save_dir="data"):
    if uploaded_file is not None:
        save_path = os.path.join(save_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return save_path
    return None

def main():
    st.set_page_config(page_title="Ask Chatbot!", layout="wide")
    
    st.markdown(
        """
        <style>
        .stApp {background-color: #181825;}
        .block-container {padding-top: 2rem;}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style='padding: 1em; background-color: #232634; border-radius: 10px; margin-bottom: 1em;'>
            <h2>🤖 <span style='color:#FF4B4B;'>Medi-Bot</span></h2>
            <p>Upload your PDFs on the left. Ask anything and get answers with sources!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.title("🤖 Medi-Bot!")

    st.markdown(
        """
        <div style='padding: 1em; background-color: #232634; border-radius: 10px; margin-bottom: 1em;'>
            <h3>👋 Welcome to <span style='color:#FF4B4B;'>Medi-Bot</span>!</h3>
            <p>Upload your medical PDFs on the left. Ask any question and get answers with sources!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Only rebuild if the flag is set
    if st.session_state.get("rebuild_vectorstore", False):
        with st.spinner("Updating knowledge base..."):
            rebuild_vectorstore()
            st.cache_resource.clear()
        st.session_state.rebuild_vectorstore = False
        st.success("Knowledge base updated! You can now search your new PDF.")

    # Sidebar for upload and file display
    with st.sidebar:
        st.header("📚 Knowledge Base Manager")
        uploaded_file = st.file_uploader("Upload a PDF to add to your knowledge base", type=["pdf"])
        if uploaded_file is not None:
            os.makedirs("data", exist_ok=True)
            save_path = save_uploaded_file(uploaded_file, "data")
            if save_path:
                st.success(f"Uploaded and saved: {uploaded_file.name}")
                st.session_state.rebuild_vectorstore = True  # Set flag

        st.markdown("---")
        st.subheader("Current files in your knowledge base:")
        data_files = [f for f in os.listdir("data") if f.lower().endswith(".pdf")]
        if data_files:
            for file in data_files:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"- {file}")
                with col2:
                    if st.button("🗑️", key=f"delete_{file}"):
                        os.remove(os.path.join("data", file))
                        st.success(f"Deleted: {file}")
                        st.session_state.rebuild_vectorstore = True  # Set flag
                        st.rerun()
        else:
            st.markdown("_No PDF files found in the data folder._")

    # Main chat area
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you dont know the answer, just say that you dont know, dont try to make up an answer. 
        Dont provide anything out of the given context

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]

            # Show the answer
            st.chat_message('assistant').markdown(f"**Answer:**\n{result}")

            # Show sources in a neat list
            if source_documents:
                st.markdown("**Sources:**")
                for i, doc in enumerate(source_documents, 1):
                    source_info = []
                    if "source" in doc.metadata:
                        source_info.append(f"File: `{doc.metadata['source']}`")
                    if "page" in doc.metadata:
                        source_info.append(f"Page: {doc.metadata['page']}")
                    st.markdown(f"{i}. {' | '.join(source_info)}")
            else:
                st.markdown("_No sources found._")

            # Save to chat history
            st.session_state.messages.append({'role': 'assistant', 'content': f"**Answer:**\n{result}"})

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()