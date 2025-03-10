import streamlit as st
import tiktoken
import os
import glob
from loguru import logger
import pickle

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

VECTORSTORE_PATH = "mediba_vectorstore.pkl"  # 저장할 벡터스토어 파일 경로

def main():
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
        #uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")
    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        if not os.path.exists(VECTORSTORE_PATH):
            directory = "C:/Users/ASUS/test pdf"
            files_text = load_local_files(directory)
            #files_text = get_text(uploaded_files)
            text_chunks = get_text_chunks(files_text)
            vetorestore = get_vectorstore(text_chunks)
            save_vectorstore(vetorestore, VECTORSTORE_PATH)
        else:
            vetorestore = load_vectorstore(VECTORSTORE_PATH)  # 벡터스토어 불러오기
     
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) 

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    


# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

#로컬 문서 활용 함수
def load_local_files(directory):
    # 지정한 디렉토리에서 pdf, docx, pptx 파일을 모두 검색
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
        logger.info(f"Loaded {file}")
    
    return documents


def get_text(docs):

    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def save_vectorstore(vectordb, path):
    with open(path, 'wb') as f:
        pickle.dump(vectordb, f)
    logger.info(f"Vectorstore saved at {path}")

def load_vectorstore(path):
    with open(path, 'rb') as f:
        vectordb = pickle.load(f)
    logger.info(f"Vectorstore loaded from {path}")
    return vectordb

def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, 
                    model_name = 'gpt-4o',
                    temperature=0,
                    max_tokens=900)
    
    system_template = """
    당신은 주어진 문서를 바탕으로 상세하고 정확한 정보를 제공하는 AI 어시스턴트입니다. 사용자의 질문에 답변할 때 다음 지침을 따라주세요.
    
    1. **문서 내용 우선**: 문서의 정보를 가장 우선으로 삼아 답변하세요. 문서에서 답변을 찾을 수 없는 경우에만 일반적인 정보를 보완하세요.
    2. **직접 인용 및 근거 제공**: 가능한 한 문서의 표현을 직접 사용하여 인용하고, 출처가 되는 문서의 특정 섹션이나 위치를 언급하여 신뢰성을 높이세요.
    3. **세분화된 청크 제공**: 문서에서 정보를 가져올 때는 각 정보를 청크 단위로 나누어, 질문과 관련된 핵심 정보를 단계별로 설명하세요.
    4. **추가 설명 제공**: 문서에서 유도된 내용이 사용자에게 명확히 전달되도록 필요한 경우 추가 설명을 제공하세요. 그러나 이 추가 설명은 문서 내용과 일관성을 유지하도록 주의하세요.

    **목표**: 문서의 구체적인 내용과 표현을 활용하여, 사용자의 질문에 정확한 정보를 제공하며 문서 기반의 신뢰성 높은 답변을 제시하는 것입니다.
    """
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain



if __name__ == '__main__':
    main()