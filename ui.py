import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model = OpenAIEmbeddings()
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm():
    llm = ChatOpenAI(
        temperature=0.5,
        model_name="gpt-3.5-turbo",  # or "gpt-4" if you have access
        max_tokens=512
    )
    return llm




def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt=st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

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
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
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
                    # Show file name and page number if available
                    source_info = []
                    if "source" in doc.metadata:
                        source_info.append(f"File: `{doc.metadata['source']}`")
                    if "page" in doc.metadata:
                        source_info.append(f"Page: {doc.metadata['page']}")
                    st.markdown(f"{i}. {' | '.join(source_info)}")
            else:
                st.markdown("_No sources found._")

            # Save to chat history
            st.session_state.messages.append({'role':'assistant', 'content': f"**Answer:**\n{result}"})

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()