# Імпорти для завантаження та обробки документів
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Імпорти для роботи з Ollama та базою даних
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Імпорти для логіки ланцюжків та промптів
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# 1. Локальні ембеддінги (теж через Ollama)
# Вони перетворять ваш текст на вектори прямо на вашому CPU/GPU
#embeddings = OllamaEmbeddings(model="llama3.2")

# 2. Локальна модель
llm = OllamaLLM(
    model="aya-expanse",
    temperature=0.1,
    num_ctx=8192,
    streaming=True
)

# 3. Ваш кастомний промпт (залишається без змін)
template = """Ти мій особистий помічник. 
Відповідай виключно українською мовою (або тією мовою, якою написано питання).
Відповідай, ґрунтуючись на моїх щоденних записах, використовуючи дружній, але прямий тон.

Контекст із щоденників:
{context}

Питання: {question}
Відповідь:"""

CUSTOM_PROMPT = PromptTemplate(
    template=template, 
    input_variables=["context", "question"]
)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


# 4. Створюємо той самий 'retriever', якого не вистачало
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# ТЕПЕР ВАШ КОД ЗАПРАЦЮЄ:
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever, 
    chain_type_kwargs={"prompt": CUSTOM_PROMPT}
)

# 3. Функція для діалогу
def start_chat():
    print("\n" + "="*50)
    print("ljbot готовий")
    print("="*50 + "\n")

    while True:
        query = input("👤 Ви: ")
        if query.lower() in ['вихід', 'exit', 'quit', '']:
            break
        
        print("\n⏳ Думаю...")
        try:
            # Використовуємо .invoke для сумісності з новими версіями
            response = qa_chain.invoke({"query": query})
            print(f"\n🤖 Бот: {response['result']}\n")
            print("-" * 30)
        except Exception as e:
            print(f"❌ Помилка: {e}")

if __name__ == "__main__":
    start_chat()