import time
import streamlit as st
import tempfile
from dotenv import load_dotenv

# LangChain 및 연동 라이브러리
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools.retriever import create_retriever_tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# 프로젝트/특수 목적 라이브러리
from langchain_teddynote import logging
from langchain_teddynote.messages import random_uuid

from typing import Annotated
from typing_extensions import TypedDict


# PaperState 정의
class PaperState(TypedDict):
    path: Annotated[str, "FilePath"]
    name: Annotated[str, "FileName"]
    question: Annotated[str, "Question"]
    context: Annotated[str, "Context"]
    answer: Annotated[str, "Answer"]
    messages: Annotated[list, "Messages"]


# PDF 파일에서 벡터스토어 기반 retriever 생성
def create_retriever(file_path):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()

    pc = Pinecone()
    index_name = "paper"
    index_names = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in index_names:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"{index_name} 인덱스를 새로 생성했습니다.")
    else:
        print(
            f"{index_name} 인덱스가 이미 존재합니다. 기존 인덱스에 데이터를 추가합니다."
        )

    vectorstore = PineconeVectorStore.from_documents(
        split_documents, embeddings, index_name=index_name
    )
    print("문서가 인덱스에 추가되었습니다.")
    return vectorstore.as_retriever()


# Chat 노드
def chat_node(state: PaperState) -> PaperState:
    question = state["question"]
    file_name = state["name"]
    file_path = state["path"]
    pdf_retriever = create_retriever(file_path)
    retriever_tool = create_retriever_tool(
        retriever=pdf_retriever,
        name="pdf_retriever",
        description="Search and return information about PDF file.",
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_tools = llm.bind_tools([retriever_tool])
    prompt = PromptTemplate.from_template(
        """제공된 문서를 참고해서 질문에 대답해줘. 
한국말로 대답해줘.
# 제공된 문서 이름
{file_name}

# 질문:
{question}"""
    )
    chain = prompt | llm_tools
    answer = chain.invoke({"file_name": file_name, "question": question})
    return {"messages": [answer], "answer": answer}


# Retrieval 노드
def retrieval_node(state: PaperState) -> PaperState:
    question = state["question"]
    file_path = state["path"]
    pdf_retriever = create_retriever(file_path)
    context = pdf_retriever.invoke(question)
    return {"context": context}


# Generation 노드
def generation_node(state: PaperState) -> PaperState:
    question = state["question"]
    context = state["context"]
    file_name = state["name"]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate.from_template(
        """제공된 문서를 참고해서 질문에 대답해줘. 
참고 내용의 페이지가 있으면 알려줘.
한국말로 대답해줘.
# 제공된 문서 이름
{file_name}

# 참고 내용
{context}

# 질문:
{question}"""
    )
    chain = prompt | llm
    answer = chain.invoke(
        {"file_name": file_name, "context": context, "question": question}
    )
    return {"answer": answer}


# 그래프 생성 함수
def create_graph():
    graph_builder = StateGraph(PaperState)
    graph_builder.add_node("chatbot", chat_node)
    graph_builder.add_node("retriever", retrieval_node)
    graph_builder.add_node("generator", generation_node)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
        {
            "tools": "retriever",
            END: END,
        },
    )
    graph_builder.add_edge("retriever", "generator")
    graph_builder.add_edge("generator", END)
    return graph_builder.compile(checkpointer=MemorySaver())


def main():
    load_dotenv()
    logging.langsmith(project_name="Paper-bot")

    st.set_page_config(page_title="Chatbot", page_icon="🤖")
    st.title("🤖 Chatbot")
    st.write("기본 LLM 챗봇")

    with st.sidebar:
        uploaded_file = st.file_uploader(
            "Upload your file", type=["pdf"], accept_multiple_files=False
        )

    if not uploaded_file:
        with st.chat_message("assistant"):
            st.markdown("논문 PDF 파일을 입력해주세요.")
        return

    # 파일 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_name = tmp_file.name

    inputs = PaperState(
        name=uploaded_file.name,
        path=tmp_file_name,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "무엇을 도와드릴까요?"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("무엇이든 물어보세요"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            config = RunnableConfig(
                recursion_limit=5, configurable={"thread_id": random_uuid()}
            )

            inputs["question"] = user_input
            graph = create_graph()
            outputs = graph.invoke(inputs, config=config)
            chatbot_response = outputs["answer"].content

            # 타이핑 효과 출력
            for char in chatbot_response:
                full_response += char
                time.sleep(0.03)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )


if __name__ == "__main__":
    main()
