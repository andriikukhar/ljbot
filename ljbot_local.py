# Імпорти для завантаження та обробки документів
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Імпорти для роботи з Ollama та базою даних
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Імпорти для логіки ланцюжків та промптів
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# 1. Локальні ембеддінги (теж через Ollama)
# Вони перетворять ваш текст на вектори прямо на вашому CPU/GPU
#embeddings = OllamaEmbeddings(model="llama3.2")

# 2. Локальна модель
llm = OllamaLLM(
    model="gemma2:9b",
    temperature=0.1
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


# 1. Завантажуємо документи (якщо ви їх не завантажили раніше в цьому ж файлі)
loader = DirectoryLoader('./gdocs/', glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# 2. Створюємо локальні ембеддінги
embeddings = OllamaEmbeddings(model="llama3.2")

# 3. Створюємо векторну базу (або завантажуємо існуючу локальну)
vector_store = Chroma(
    persist_directory="./chroma_db_local", 
    embedding_function=embeddings
)

# 4. Створюємо той самий 'retriever', якого не вистачало
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ТЕПЕР ВАШ КОД ЗАПРАЦЮЄ:
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever, 
    chain_type_kwargs={"prompt": CUSTOM_PROMPT}
)

# 1. Ініціалізація пошуковика (Retriever)
# k=3 означає, що бот знайде 3 найбільш схожі фрагменти з ваших записів
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 2. Створення ланцюжка RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": CUSTOM_PROMPT}
)

# 3. Функція для діалогу
def start_chat():
    print("\n" + "="*50)
    print("🤖 ЛОКАЛЬНИЙ БОТ-ПОМІЧНИК ГОТОВИЙ")
    print("Я читаю ваші щоденники через Ollama")
    print("Напишіть 'вихід', щоб завершити.")
    print("="*50 + "\n")

    while True:
        query = input("👤 Ви: ")
        if query.lower() in ['вихід', 'exit', 'quit']:
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