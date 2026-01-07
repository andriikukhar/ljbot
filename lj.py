import os
import re
import ssl
import shutil
from datetime import datetime
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# 1. Налаштування безпеки та середовища
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 2. Функція логічного розбиття: ОДИН ДЕНЬ = ОДИН ДОКУМЕНТ
def split_by_diary_entry(documents):
    diary_entries = []
    # Регулярний вираз для двох форматів:
    # 1. [дек. 31, 2008|03:20 pm] - (враховуючи укр/рос місяці та час)
    # 2. 5.07.2019 чи 05.07.2019
    date_pattern = r'(\[\w{1,4}\.?\s\d{1,2},\s\d{4}\|.*?\]|\d{1,2}\.\d{2}\.\d{4})'
    
    for doc in documents:
        content = doc.page_content
        parts = re.split(date_pattern, content)
        
        # parts[0] - текст до першої дати, ігноруємо якщо там порожньо
        for i in range(1, len(parts), 2):
            date_str = parts[i]
            entry_text = parts[i+1] if (i+1) < len(parts) else ""
            
            if len(entry_text.strip()) > 5:
                full_content = f"Дата: {date_str}\n{entry_text.strip()}"
                diary_entries.append(Document(
                    page_content=full_content,
                    metadata={"date": date_str, "source": doc.metadata.get("source")}
                ))
    return diary_entries

# 3. Завантаження та індексація
print("Wait... Оновлюю базу знань...")
loader = DirectoryLoader('./gdocs/', glob="**/*.txt", loader_cls=TextLoader)
raw_documents = loader.load()
texts = split_by_diary_entry(raw_documents)

# Видаляємо стару базу, щоб переіндексувати нові "шматки" (опціонально)
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", api_key=os.environ["GEMINI_API_KEY"])
vector_store = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")

# 4. Налаштування моделі та ретрівера
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", # Рекомендую 2.0 за швидкість і контекст
    temperature=0.1,
    api_key=os.environ["GEMINI_API_KEY"]
)

retriever = vector_store.as_retriever(
    search_type="mmr", # Різноманітність результатів
    search_kwargs={"k": 15, "fetch_k": 40} 
)

# 5. Промпт та Ланцюжок
current_date_str = datetime.now().strftime("%d.%m.%Y")

template = """Ти мій особистий цифровий двійник-помічник. 
Сьогоднішня дата: {current_date}. 

Твоє завдання: відповідати на питання, базуючись на моїх щоденниках. 
Якщо питання стосується періоду (наприклад, "що було минулого літа"), проаналізуй дати в наданих фрагментах.

Контекст із щоденників:
{context}

Питання: {question}
Відповідь:"""

CUSTOM_PROMPT = PromptTemplate(
    template=template, 
    input_variables=["context", "question"],
    partial_variables={"current_date": current_date_str}
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": CUSTOM_PROMPT}
)

# 6. Чат
def run_chat_bot(chain):
    print(f"\n🤖 Бот активовано! Знаю про {len(texts)} днів з твого життя.")
    while True:
        user_input = input("Твоє питання (або Enter для виходу): ")
        if not user_input:
            break

        try:
            response = chain.invoke({"query": user_input})
            print(f"\n✅ {response['result']}\n")
        except Exception as e:
            print(f"❌ Помилка: {e}")

if __name__ == "__main__":
    run_chat_bot(qa_chain)