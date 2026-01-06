import os
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ssl
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Завантажуємо всі файли .txt з папки 'journals/'
loader = DirectoryLoader('./gdocs/', glob="**/*.docx")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # Оптимальний розмір фрагмента
    chunk_overlap=200 # Перекриття для збереження контексту
)
texts = text_splitter.split_documents(documents)

embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", api_key=os.environ["GEMINI_API_KEY"])

# Створення векторної бази даних і зберігання
vector_store = Chroma.from_documents(
    texts,
    embeddings,
    persist_directory="./chroma_db" # Зберігаємо базу локально
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2, # Низька температура для фактичних відповідей
    api_key=os.environ["GEMINI_API_KEY"]
)

# Ініціалізація пошуковика (Retriever)
retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Шукаємо 3 найбільш релевантних фрагменти

# Створення ланцюжка RAG
# LlamaIndex або Langchain можуть автоматично побудувати запит
# 'question-answering' до моделі, додаючи знайдений контекст
qa_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type="stuff", # Кладе весь знайдений контекст у промпт
    retriever=retriever
)

def run_chat_bot(qa_chain):
    print("🤖 Бот пам'яті активовано! Запитуйте про ваші щоденники.")
    while True:
        user_input = input("Яке питання? (або 'вихід'): ")
        if user_input.lower() == 'вихід':
            print("До зустрічі!")
            break

        # Виконуємо запит RAG
        result = qa_chain.run(user_input)
        print(f"\n✅ Відповідь: {result}\n")


template = """Ти мій особистий помічник. Відповідай, ґрунтуючись на моїх щоденних записах, використовуючи дружній, але прямий тон.

Контекст із щоденників:
{context}

Питання: {question}
Відповідь:"""

CUSTOM_PROMPT = PromptTemplate(
    template=template, 
    input_variables=["context", "question"]
)

# 2. Передаємо цей промпт у ланцюжок через chain_type_kwargs
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": CUSTOM_PROMPT}  # Оце ключовий рядок
)

# Запуск
run_chat_bot(qa_chain)
