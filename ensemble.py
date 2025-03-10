import streamlit as st
import tiktoken
import os
import glob
import pickle
import hashlib
from loguru import logger
from langchain.prompts import PromptTemplate

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

VECTORSTORE_PATH = "mediba_vectorstore.pkl"  # 벡터스토어 파일 경로

def get_file_hash(file_path):
    """파일의 MD5 해시를 계산하여 변경 사항을 감지합니다."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_file_hashes(hash_file_path):
    """이전에 저장된 파일 해시를 불러옵니다."""
    try:
        with open(hash_file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def save_file_hashes(file_hashes, hash_file_path):
    """현재 파일 해시를 저장합니다."""
    with open(hash_file_path, 'wb') as f:
        pickle.dump(file_hashes, f)

def tiktoken_len(text):
    """텍스트의 토큰 길이를 계산합니다."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def update_vectorstore(directory, existing_vectorstore_path, hash_file_path='file_hashes.pkl'):
    """새로운 또는 수정된 문서로 벡터 스토어를 업데이트합니다."""
    try:
        vectorstore, all_documents = load_vectorstore(existing_vectorstore_path)
    except FileNotFoundError:
        vectorstore = None
        all_documents = []

    file_hashes = load_file_hashes(hash_file_path)
    
    files = glob.glob(os.path.join(directory, "*.pdf")) + \
            glob.glob(os.path.join(directory, "*.docx")) + \
            glob.glob(os.path.join(directory, "*.pptx"))
    
    new_documents = []
    updated_hashes = file_hashes.copy()

    for file in files:
        current_hash = get_file_hash(file)
        if file not in file_hashes or file_hashes[file] != current_hash:
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file)
            elif file.endswith('.docx'):
                loader = Docx2txtLoader(file)
            elif file.endswith('.pptx'):
                loader = UnstructuredPowerPointLoader(file)
            
            file_documents = loader.load_and_split()
            new_documents.extend(file_documents)
            updated_hashes[file] = current_hash
            logger.info(f"새로운/수정된 파일 처리: {file}")
    
    if not new_documents:
        logger.info("새로운 또는 수정된 문서가 없습니다.")
        return vectorstore, all_documents
    
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    new_chunks = text_splitter.split_documents(new_documents)
    all_documents.extend(new_documents)
    
    if vectorstore is None:
        vectorstore = FAISS.from_documents(new_chunks, embeddings)
    else:
        vectorstore.add_documents(new_chunks)
    
    save_vectorstore((vectorstore, all_documents), existing_vectorstore_path)
    save_file_hashes(updated_hashes, hash_file_path)
    
    logger.info(f"{len(new_documents)}개의 새로운/수정된 문서로 벡터 스토어 업데이트")
    return vectorstore, all_documents

def save_vectorstore(data, path):
    """벡터 스토어와 문서를 파일로 저장합니다."""
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"벡터 스토어 저장 위치: {path}")

def load_vectorstore(path):
    """저장된 벡터 스토어와 문서를 불러옵니다."""
    with open(path, 'rb') as f:
        vectorstore, all_documents = pickle.load(f)
    logger.info(f"벡터 스토어 로드 위치: {path}")
    return vectorstore, all_documents

def get_conversation_chain(vectorstore, all_documents, openai_api_key):
    """대화형 검색 체인을 생성합니다."""
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4o', temperature=0, max_tokens=900)
    
    bm25_retriever = BM25Retriever.from_documents(all_documents)
    faiss_retriever = vectorstore.as_retriever(search_type='mmr', verbose=True)
    
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever])
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=ensemble_retriever,
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )

    return conversation_chain

def main():
    st.set_page_config(page_title="DirChat", page_icon=":books:")
    st.title("_Private Data :red[QA Chat]_ :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")
    
    if process:
        if not openai_api_key:
            st.info("OpenAI API 키를 추가해주세요.")
            st.stop()

        vectorstore = update_vectorstore(
            directory="C:/Users/ASUS/test pdf", 
            existing_vectorstore_path=VECTORSTORE_PATH
        )
     
        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key) 
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation
            with st.spinner("생각 중..."):
                result = chain({"question": query})
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    for idx, doc in enumerate(source_documents[:5], 1):
                        st.markdown(f"{idx}. {doc.metadata['source']}", help=doc.page_content)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()
