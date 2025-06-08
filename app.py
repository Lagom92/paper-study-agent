import time
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_teddynote import logging
from dotenv import load_dotenv


def main():
    # .env 파일에서 환경변수 로드 (API Key 등)
    load_dotenv()

    # LangSmith 프로젝트 로깅 설정 (분석 및 추적용)
    logging.langsmith(project_name="Paper-bot")

    # Streamlit 페이지 기본 설정
    st.set_page_config(page_title="Chatbot", page_icon="🤖")
    st.title("🤖 Chatbot")
    st.write("기본 LLM 챗봇")

    # 세션 상태에 메시지 이력이 없으면 초기화 (첫 메시지: 챗봇 인사)
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "무엇을 도와드릴까요?"}
        ]

    # 이전 대화 메시지 화면에 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 받기
    if user_input := st.chat_input("무엇이든 물어보세요"):
        # 사용자 메시지 세션 상태에 저장 및 화면에 출력
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 챗봇 응답 생성 및 출력
        with st.chat_message("assistant"):
            message_placeholder = st.empty()  # 실시간 응답 출력을 위한 placeholder
            full_response = ""

            # 프롬프트 템플릿 정의
            prompt = PromptTemplate.from_template(
                """너는 사용자의 질문에 대답하는 챗봇이야. 
            한국말로 대답해줘.

            #Question:
            {question}

            #Answer:"""
            )

            # LLM 객체 생성 (OpenAI GPT-3.5-turbo)
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

            # LCEL 체인 구성: 입력 → 프롬프트 → LLM → 문자열 파싱
            chain = (
                {"question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
            )

            # 체인 실행하여 응답 생성
            assistant_response = chain.invoke(user_input)

            # 응답을 한 글자씩 출력하여 타이핑 효과 구현
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        # 챗봇의 응답을 세션 상태에 저장
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )


if __name__ == "__main__":
    main()
