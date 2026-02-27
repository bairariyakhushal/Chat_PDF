import boto3
import os
import json
import tempfile
import streamlit as st

from langchain_aws import BedrockEmbeddings
from langchain_aws import ChatBedrockConverse

import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

## Bedrock Client and Embedding
client=boto3.client(service_name="bedrock-runtime",region_name="us-east-1")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=client)

## Data ingestion
def data_ingestion(uploaded_files):
    docs=[]
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300
    )
    
    for uploaded_file in uploaded_files:
        # Temp file mein save karo kyunki PyPDFLoader ko file path chahiye
        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path=tmp_file.name
            
            
        loader=PyPDFLoader(tmp_path)
        documents=loader.load()
        chunks=splitter.split_documents(documents)
        docs.extend(chunks)
        
        os.unlink(tmp_path)
    
    return docs
    
## Vector store banana
def get_vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(docs,bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")
    return vectorstore_faiss

def get_nova_micro_llm():
    llm=ChatBedrockConverse(
        model_id="amazon.nova-micro-v1:0",
        client=client,
        max_tokens=512,
        temperature=0.5
    )
    return llm

def get_llama_llm():
    llm=ChatBedrockConverse(
        model_id="us.meta.llama3-1-8b-instruct-v1:0",
        client=client,
        max_tokens=512,
        temperature=0.5
    )
    return llm
    
prompt_template="""
Use the following pieces of context to provide a 
concise answer to the question at the end but use atleast 250 words 
with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

"""

prompt=PromptTemplate(
    template=prompt_template,
    input_variables=["context","question"]
)

def get_response_llm(llm,vectorstore_faiss,query):
    qa=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    answer=qa.invoke({"query":query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with your PDFs using AWS Bedrock 💁")

    ## Session state mein faiss index store karo taaki baar baar load na karna pade
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None

    with st.sidebar:
        st.title("Upload Your PDFs")
        
        uploaded_files = st.file_uploader(
            "PDF files choose karo", 
            type="pdf", 
            accept_multiple_files=True  # multiple PDFs ek saath
        )
        
        if st.button("Process PDFs"):
            if not uploaded_files:
                st.warning("Upload PDF first!")
            else:
                with st.spinner("Processing..."):
                    docs = data_ingestion(uploaded_files)
                    st.session_state.faiss_index = get_vector_store(docs)
                    st.success(f"{len(uploaded_files)} PDF(s) successfully processed!")

    user_question = st.text_input("Ask Your Question")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Nova Micro Output"):
            if st.session_state.faiss_index is None:
                st.warning("First Process the PDF!")
            elif not user_question:
                st.warning("First Ask any question!")
            else:
                with st.spinner("Processing..."):
                    llm = get_nova_micro_llm()
                    st.write(get_response_llm(llm, st.session_state.faiss_index, user_question))
                    st.success("Done!")

    with col2:
        if st.button("Llama 3.1 Output"):
            if st.session_state.faiss_index is None:
                st.warning("First Process the PDF!")
            elif not user_question:
                st.warning("First Ask any question!")
            else:
                with st.spinner("Processing..."):
                    llm = get_llama_llm()
                    st.write(get_response_llm(llm, st.session_state.faiss_index, user_question))
                    st.success("Done!")

if __name__ == "__main__":
    main()