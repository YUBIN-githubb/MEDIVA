import streamlit as st
import tiktoken
import os
import glob
import pickle
import hashlib
from loguru import logger

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
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain

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

def update_vectorstore(directory, existing_vectorstore_path, hash_file_path='file_hashes.pkl'):
    """
    새로운 또는 수정된 문서로 벡터 스토어를 업데이트합니다.
    
    인자:
    - directory: 문서가 포함된 디렉토리 경로
    - existing_vectorstore_path: 기존 벡터 스토어 경로
    - hash_file_path: 파일 해시 저장 경로
    
    반환:
    - 업데이트된 FAISS 벡터 스토어
    """
    # 기존 벡터 스토어 로드
    try:
        vectorstore = load_vectorstore(existing_vectorstore_path)
    except FileNotFoundError:
        vectorstore = None

    # 기존 파일 해시 로드
    file_hashes = load_file_hashes(hash_file_path)
    
    # 지원되는 문서 파일 검색
    files = glob.glob(os.path.join(directory, "*.pdf")) + \
            glob.glob(os.path.join(directory, "*.docx")) + \
            glob.glob(os.path.join(directory, "*.pptx"))
    
    # 추가할 새 문서 추적
    new_documents = []
    updated_hashes = file_hashes.copy()

    for file in files:
        current_hash = get_file_hash(file)
        
        # 파일이 새로운지 또는 수정되었는지 확인
        if file not in file_hashes or file_hashes[file] != current_hash:
            # 파일 로드
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file)
            elif file.endswith('.docx'):
                loader = Docx2txtLoader(file)
            elif file.endswith('.pptx'):
                loader = UnstructuredPowerPointLoader(file)
            
            # 문서 로드 및 분할
            file_documents = loader.load_and_split()
            new_documents.extend(file_documents)
            
            # 해시 업데이트
            updated_hashes[file] = current_hash
            logger.info(f"새로운/수정된 파일 처리: {file}")
    
    # 새 문서가 없으면 기존 벡터 스토어 반환
    if not new_documents:
        logger.info("새로운 또는 수정된 문서가 없습니다.")
        return vectorstore
    
    # 새 문서 임베딩
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 새 문서를 청크로 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    new_chunks = text_splitter.split_documents(new_documents)
    
    # 벡터 스토어 업데이트 또는 생성
    if vectorstore is None:
        vectorstore = FAISS.from_documents(new_chunks, embeddings)
    else:
        # 기존 벡터 스토어에 새 문서 추가
        vectorstore.add_documents(new_chunks)
    
    # 업데이트된 벡터 스토어 저장
    save_vectorstore(vectorstore, existing_vectorstore_path)
    
    # 업데이트된 파일 해시 저장
    save_file_hashes(updated_hashes, hash_file_path)
    
    logger.info(f"{len(new_documents)}개의 새로운/수정된 문서로 벡터 스토어 업데이트")
    
    return vectorstore

def tiktoken_len(text):
    """텍스트의 토큰 길이를 계산합니다."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def load_local_files(directory):
    """
    지정된 디렉토리에서 pdf, docx, pptx 파일을 로드합니다.
    
    인자:
    - directory: 문서가 포함된 디렉토리 경로
    
    반환:
    - 로드된 문서 목록
    """
    files = glob.glob(os.path.join(directory, "*.pdf")) + \
            glob.glob(os.path.join(directory, "*.docx")) + \
            glob.glob(os.path.join(directory, "*.pptx"))
    
    documents = []
    for file in files:
        if file.endswith('.pdf'):
            loader = PyPDFLoader(file)
        elif file.endswith('.docx'):
            loader = Docx2txtLoader(file)
        elif file.endswith('.pptx'):
            loader = UnstructuredPowerPointLoader(file)
        
        # 파일 내용 불러오기
        file_documents = loader.load_and_split()
        documents.extend(file_documents)
        logger.info(f"로드됨: {file}")
    
    return documents

def save_vectorstore(vectordb, path):
    """벡터 스토어를 파일로 저장합니다."""
    with open(path, 'wb') as f:
        pickle.dump(vectordb, f)
    logger.info(f"벡터 스토어 저장 위치: {path}")

def load_vectorstore(path):
    """저장된 벡터 스토어를 불러옵니다."""
    with open(path, 'rb') as f:
        vectordb = pickle.load(f)
    logger.info(f"벡터 스토어 로드 위치: {path}")
    return vectordb

def get_conversation_chain(vetorestore, openai_api_key):
    """대화형 검색 체인을 생성합니다."""
    llm = ChatOpenAI(openai_api_key=openai_api_key, 
                    model_name = 'gpt-4o',
                    temperature=0,
                    max_tokens=2000)
    
    system_template = """
    당신은 주어진 문서를 바탕으로 상세하고 정확한 정보를 제공하는 AI 어시스턴트입니다. 사용자의 질문에 답변할 때 다음 지침을 따라주세요.
    다음 맥락(context)을 참고하여 질문에 답변해주세요:
    {context}
    사용자의 질문에 답변할 때 다음 지침을 따라주세요:
    1. **문서 내용 우선**: 문서의 정보를 가장 우선으로 삼아 답변하세요. 문서에서 답변을 찾을 수 없는 경우에만 일반적인 정보를 보완하세요.
    2. **직접 인용 및 근거 제공**: 가능한 한 문서의 표현을 직접 사용하여 인용하고, 출처가 되는 문서의 특정 섹션이나 위치를 언급하여 신뢰성을 높이세요.
    3. **세분화된 청크 제공**: 문서에서 정보를 가져올 때는 각 정보를 청크 단위로 나누어, 질문과 관련된 핵심 정보를 단계별로 설명하세요.
    4. **추가 설명 제공**: 문서에서 유도된 내용이 사용자에게 명확히 전달되도록 필요한 경우 추가 설명을 제공하세요. 그러나 이 추가 설명은 문서 내용과 일관성을 유지하도록 주의하세요.
    5. **관련 예시와 추가 정보 제공**: 정보를 쉽게 이해할 수 있도록 실제 사례나 비슷한 개념을 참고한 문서 기반으로 제시하세요.
    6. **사용자의 질문 확장**: 질문이 모호하면 참고한 문서를 바탕으로 관련된 주제를 더 다루고, 추가 정보를 제공하세요.
    7. **정보 구조화**: 주요 내용을 요약한 후 상세한 내용을 나열하세요.
    **목표**: 문서의 구체적인 내용과 표현을 활용하여, 사용자의 질문에 정확한 정보를 제공하며 문서 기반의 신뢰성 높은 답변을 제시하는 것입니다.
    질문: {question}
    """

    prompt = PromptTemplate(
        template=system_template, 
        input_variables=["context", "question"]
    )

    
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    

    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type='similarity', verbose=True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose=True,
            combine_docs_chain_kwargs={
                'prompt': prompt,
                'document_variable_name': 'context'
            }  # 여기에 chat_prompt 추가
        )

    return conversation_chain

def main():
    """메인 스트림릿 애플리케이션 함수"""
    st.set_page_config(
    page_title="DirChat",
    page_icon=":books:")

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

        # 새로운 update_vectorstore 함수 사용
        vetorestore = update_vectorstore(
            directory="C:/Users/ASUS/test pdf", 
            existing_vectorstore_path=VECTORSTORE_PATH
        )
     
        st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key) 

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 의료기기 임상시험 계획서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # 채팅 로직
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("생각 중..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                
                with st.expander("참고 문서 확인"):
                    for idx, doc in enumerate(source_documents[:6], 1):
                        page = doc.metadata.get('page', 'N/A')  # 페이지 정보가 있는 경우 가져옴
                        st.markdown(f"{idx}. {doc.metadata['source']}  {page}페이지", help=doc.page_content)
                        


        # 챗봇 메시지를 채팅 기록에 추가
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()