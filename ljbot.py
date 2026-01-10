import os
import re
import shutil
from datetime import datetime
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# --- 1. ЛОГІКА ОБРОБКИ (ОДИН ДЕНЬ = ОДИН ЦІЛІСНИЙ БЛОК) ---
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

# --- 2. ПІДГОТОВКА ДАНИХ ---
print("🔍 Аналіз архівів (Режим Long Context + Memory)...")
loader = DirectoryLoader('./gdocs/', glob="**/*.txt", loader_cls=TextLoader)
raw_docs = loader.load()
texts = split_diary_by_entries(raw_docs)

if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
vector_store = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")

# --- 3. МОДЕЛЬ ТА ПОШУК ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.2,
    max_output_tokens=2048
)

retriever = vector_store.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 50, "fetch_k": 100, "lambda_mult": 0.3}
)

# --- 4. НОВИЙ ПІДХІД ДО ПАМ'ЯТІ ---
# Зберігаємо історію чатів у словнику (можна використати БД для продакшену)
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

today = datetime.now().strftime("%d.%m.%Y")

# Новий промпт з MessagesPlaceholder для історії
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

# Створюємо ланцюжки
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Обгортаємо з історією повідомлень
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# --- 5. ЗАПУСК ---
def run_smart_bot_with_memory(chain):
    print(f"✅ База знань готова ({len(texts)} днів). Пам'ять активована.")
    session_id = "user_session_1"  # Можна генерувати унікальний ID для кожного користувача
    
    while True:
        user_input = input("\nЗапитай щось: ")
        if not user_input: 
            break
        
        print("🧠 Аналізую архіви та пам'ятаю нашу розмову...")
        
        res = chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        
        print(f"\n📝 ВІДПОВІДЬ:\n{res['answer']}")
        print("________________________________________________________________________________________")

if __name__ == "__main__":
    run_smart_bot_with_memory(conversational_rag_chain)