import streamlit as st
import os
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Ініціалізація
if "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]

os.environ["GOOGLE_API_KEY"] = ""

@st.cache_resource
def load_rag():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 20})

retriever = load_rag()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

today = datetime.now().strftime("%d.%m.%Y")

# Промпт
system_prompt = f"Ти — цифровий двійник. Сьогодні: {datetime.now().strftime('%d.%m.%Y')}. Використовуй контекст: {{context}}"

prompt = ChatPromptTemplate.from_messages([
    ("system", f"""Ти — професійний біограф та особистий цифровий двійник. 
Твоя суперсила — бачити зв'язки між подіями, що відбулися з різницею у роки.

Сьогоднішня дата: {today}.

ІНСТРУКЦІЇ:
1. Ретельно вивчай хронологію. Якщо питання про зміни у житті — порівнюй ранні записи з пізніми.
2. У контексті багато записів. Спочатку виділи найважливіші для відповіді дати.
3. Обов'язково цитуй конкретні дати, щоб відповідь була достовірною.
4. Якщо інформації замало для глибокого висновку, так і скажи, але запропонуй гіпотезу на основі того, що є.
5. Проаналізуй записи з щоденника та історію діалогу, щоб дати ґрунтовну відповідь. 
Якщо користувач ставить уточнююче питання (наприклад, "а чому так?"), обов'язково враховуй попередній контекст розмови.

КОНТЕКСТ ІЗ ЩОДЕННИКІВ:
{{context}}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Ланцюжок
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

if "chat_store" not in st.session_state:
    st.session_state.chat_store = {}

def get_session_history(session_id: str):
    if session_id not in st.session_state.chat_store:
        st.session_state.chat_store[session_id] = ChatMessageHistory()
    return st.session_state.chat_store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

def chat_with_bot(message, history):
    # history ігноруємо, бо LangChain сам веде пам'ять у `store`
    response = conversational_rag_chain.invoke(
        {"input": message},
        config={"configurable": {"session_id": "gradio_session"}}
    )
    return response["answer"]

st.title("🧠 Мій Персональний Бот Пам'яті")
st.caption("ШІ-асистент на основі ваших записів.")

# 1. Ініціалізація історії в Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Привіт! Я готовий зануритися у твої спогади. Про що запитаєш?"}]

# 2. Відображення історії
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 3. Обробка введення
if user_query := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.spinner("Аналізую архіви..."):
        try:
            # ВИПРАВЛЕНО: Викликаємо правильний ланцюжок
            # session_id дозволяє LangChain тримати пам'ять всередині сесії
            res = conversational_rag_chain.invoke(
                {"input": user_query},
                config={"configurable": {"session_id": "streamlit_user"}}
            )
            response = res["answer"]
        except Exception as e:
            response = f"Помилка: {e}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)