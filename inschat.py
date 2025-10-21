from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from tqdm import tqdm
import os

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

def build_index(chat_chunks):
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = ChromaVectorStore(persist_dir=DB_DIR)
    ctx = ServiceContext.from_defaults(embed_model=embed_model)
    index = VectorStoreIndex.from_documents(
        [{"text": chunk} for chunk in chat_chunks],
        service_context=ctx,
        vector_store=vector_store
    )
    index.storage_context.persist(persist_dir=DB_DIR)
    return index

def ask(index):
    print("\nReady! Type your question or 'exit':")
    while True:
        q = input("You: ")
        if q.strip().lower() in ("exit", "quit"):
            break
        res = index.as_query_engine(response_mode="compact", similarity_top_k=4)(q)
        print("\nReply:\n", res, "\n" + "-"*40)

def main():
    print("Parsing chat file ...")
    entries = parse_chat_lines(CHAT_FILE)
    print(f"Parsed {len(entries)} messages.")
    print("Chunking for indexing ...")
    chunks = chunk_entries(entries, max_len=1200)
    print(f"Created {len(chunks)} chunks.")
    print("Building vector index ...")
    index = build_index(chunks)
    ask(index)

if __name__ == "__main__":
    main()
