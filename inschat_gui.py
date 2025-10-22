import streamlit as st
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from tqdm import tqdm
from llama_index.llms.ollama import Ollama

CHAT_FILE = r"C:\Projects\InsChat\Chat\chat_converted_ist.txt"
DB_DIR = r"C:\Projects\InsChat\output\db"

def parse_chat_lines(chat_file):
    entries = []
    with open(chat_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("["):
                ts_end = line.find("]")
                if ts_end != -1:
                    ts = line[1:ts_end]
                    rest = line[ts_end+2:]
                    if ":" in rest:
                        sender, msg = rest.split(":", 1)
                        entries.append({
                            "text": msg.strip(),
                            "metadata": {"timestamp": ts, "sender": sender.strip()}
                        })
    return entries

def chunk_entries(entries, max_len=1200):
    chunks, chunk, cur_len = [], [], 0
    for entry in entries:
        line = f"[{entry['metadata']['timestamp']}] {entry['metadata']['sender']}: {entry['text']}\n"
        if cur_len + len(line) > max_len and chunk:
            chunks.append("".join(chunk))
            chunk, cur_len = [], 0
        chunk.append(line)
        cur_len += len(line)
    if chunk:
        chunks.append("".join(chunk))
    return chunks

@st.cache_resource
def build_index():
    entries = parse_chat_lines(CHAT_FILE)
    chunks = chunk_entries(entries, max_len=1200)
    
    client = chromadb.PersistentClient(path=DB_DIR)
    collection_name = "inschat"
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        collection = client.create_collection(collection_name)
    
    vector_store = ChromaVectorStore(
        chroma_collection=collection,
        client=client,
    )
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model = embed_model
    Settings.llm = Ollama(model="nous-hermes:7b", request_timeout=120)
    
    documents = [Document(text=chunk) for chunk in chunks]
    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store
    )
    return index

# Page config
st.set_page_config(page_title="Instagram Chat Assistant", page_icon="💬", layout="wide")
st.title("💬 Instagram Chat Assistant")
st.markdown("Ask questions about your Instagram chat history!")

# Load index
with st.spinner("Loading chat index..."):
    index = build_index()
    query_engine = index.as_query_engine(response_mode="compact", similarity_top_k=4)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask something about your chat history..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_engine.query(prompt)
            st.markdown(str(response))
    
    st.session_state.messages.append({"role": "assistant", "content": str(response)})

# Clear chat button in sidebar
with st.sidebar:
    st.markdown("### Controls")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
