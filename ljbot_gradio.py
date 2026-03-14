import gradio as gr
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

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
retriever = vector_store.as_retriever(search_kwargs={"k": 20})

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

store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

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

# Простий запуск без зайвих аргументів
demo = gr.ChatInterface(fn=chat_with_bot, title="LJ Bot")

if __name__ == "__main__":
    demo.launch()