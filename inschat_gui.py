import streamlit as st

from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
import chromadb

CHAT_FILE = r"C:\Projects\InsChat\Chat\chat_converted_ist.txt"
DB_DIR = r"C:\Projects\InsChat\output\db"

@st.cache_resource(show_spinner=True)
def load_index():
    client = chromadb.PersistentClient(path=DB_DIR)
    collection_name = "inschat"
    collections = [col.name for col in client.list_collections()]
    if collection_name in collections:
        collection = client.get_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection, client=client)
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Settings.llm = Ollama(model="mistral")
        return VectorStoreIndex.from_vector_store(vector_store, settings=Settings)
    else:
        # This is a basic placeholder. You should ideally use your same chunking logic here!
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            entries = [line.strip() for line in f if line.strip() and line.startswith("[")]
        documents = [Document(text=line) for line in entries]
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Settings.embed_model = embed_model
        Settings.llm = Ollama(model="mistral")
        vector_store = ChromaVectorStore(
            chroma_collection=client.create_collection(collection_name),
            client=client,
        )
        index = VectorStoreIndex.from_documents(documents, settings=Settings, vector_store=vector_store)
        return index

st.set_page_config(page_title="InsChat LLM Q&A", page_icon="💬")
st.title("InsChat Chatbot (LLM + RAG)")

index = load_index()

if "history" not in st.session_state: st.session_state["history"] = []

user_input = st.text_input("Type your question:")

if user_input:
    with st.spinner("Thinking..."):
        response = index.as_query_engine(response_mode="compact", similarity_top_k=4).query(user_input)
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", str(response)))

for speaker, text in st.session_state.history:
    st.markdown(f"**{speaker}:** {text}")
