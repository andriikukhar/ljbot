import os
import re
import time
from datetime import datetime
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

if "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]

os.environ["GOOGLE_API_KEY"] = ""

# --- 1. ЛОГІКА ОБРОБКИ ---
def split_diary_by_entries(documents):
    diary_entries = []
    date_pattern = r'(\[\w{1,4}\.?\s\d{1,2},\s\d{4}\|.*?\]|\d{1,2}\.\d{2}\.\d{4})'
    for doc in documents:
        content = doc.page_content
        parts = re.split(date_pattern, content)
        for i in range(1, len(parts), 2):
            date_str = parts[i]
            entry_text = parts[i+1] if (i+1) < len(parts) else ""
            if len(entry_text.strip()) > 10:
                full_content = f"ЗАПИС ВІД {date_str.strip('[]')}:\n{entry_text.strip()}"
                diary_entries.append(Document(
                    page_content=full_content,
                    metadata={"date": date_str, "source": doc.metadata.get("source")}
                ))
    return diary_entries

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# --- 2. ПІДГОТОВКА ДАНИХ (З ПЕРЕВІРКОЮ НАЯВНОСТІ) ---
if not os.path.exists("./chroma_db"):
    os.makedirs("./chroma_db")

# Підключаємось до бази (навіть якщо вона порожня)
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Отримуємо список того, що вже завантажено (за метаданими source або текстом)
# Це складно, тому простіший шлях — зберегти "checkpoint" у файл
checkpoint_file = "processed_idx.txt"
start_idx = 0
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r") as f:
        start_idx = int(f.read().strip())

loader = DirectoryLoader('./gdocs/', glob="**/*.txt", loader_cls=TextLoader)
texts = split_diary_by_entries(loader.load())

if start_idx < len(texts):
    print(f"⏩ Продовжуємо з індексу {start_idx} з {len(texts)}...")
    for i in range(start_idx, len(texts), 5):
        batch = texts[i : i + 5]
        try:
            vector_store.add_documents(batch)
            print(f"📦 Оброблено {i + 5} / {len(texts)}")
            # Записуємо прогрес
            with open(checkpoint_file, "w") as f:
                f.write(str(i + 5))
            time.sleep(3)
        except Exception as e:
            print(f"🛑 Ліміт! Зупиняємось на {i}. Почекай 5 хв і запусти знову.")
            break 
else:
    print("✅ Все вже завантажено!")

# --- 3. МОДЕЛЬ ТА ЛАНЦЮЖОК ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
retriever = vector_store.as_retriever(search_kwargs={"k": 20})

today = datetime.now().strftime("%d.%m.%Y")
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

# --- 4. ЗАПУСК ---
if __name__ == "__main__":
    session_id = "main_session"
    while True:
        user_input = input("\nТи: ")
        if user_input.lower() in ['exit', 'quit', 'вихід', '']: break
        
        res = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"\n🤖 Бот: {res['answer']}")
