Experimental RAG-based personal AI assistant ("digital twin") that answers questions about the user's life by searching diary/blog notes stored as `.txt` files in `./gdocs/`.

Primary bot (Google Gemini + Chroma, CLI with memory):
python ljbot.py

Data loading / building the vector store (run before first use of `ljbot.py`):
python ljbot_load.py

Gradio web UI:
python ljbot_gradio.py

Streamlit web UI:
streamlit run ljbot_streamlit.py

Local Ollama variant (no API key needed):
python ljbot_local.py       # Ollama LLM + Ollama embeddings, chroma_db_local
python ljbot_local_ge.py    # Ollama LLM (aya-expanse) + Google embeddings, chroma_db