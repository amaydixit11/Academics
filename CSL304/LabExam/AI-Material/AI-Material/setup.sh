#!/bin/bash
# ================================================
# 🔧 Offline AI Setup for Coding (Ubuntu)
# Author: Saurav's AI Assistant
# ================================================

set -e

echo "🚀 Starting Offline AI Setup..."

sudo apt update -y
sudo apt install -y curl git python3 python3-pip

# ---- Step 1: Install Ollama ----
if ! command -v ollama &> /dev/null; then
  echo "🧠 Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "✅ Ollama already installed."
fi

# ---- Step 2: Choose model ----
echo ""
echo "Which model do you want?"
echo "1. llama3.1:8b  (Fast & Good)"
echo "2. codellama:13b (Best for coding, needs 16GB+ RAM)"
read -p "Enter 1 or 2: " choice

if [ "$choice" == "2" ]; then
  MODEL="codellama:13b"
else
  MODEL="llama3.1:8b"
fi

echo "📥 Downloading model $MODEL ..."
ollama pull $MODEL

# ---- Step 3: Python env for RAG ----
echo "🐍 Installing Python packages..."
pip install --upgrade pip
pip install langchain chromadb pypdf ollama

# ---- Step 4: Folder structure ----
mkdir -p ~/offline_ai/{data,scripts}
cd ~/offline_ai

# ---- Step 5: Main Python RAG script ----
cat > scripts/ai_query.py <<EOF
import os
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

BASE_DIR = os.path.expanduser("~/offline_ai")
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "vector_db")
INPUT_FILE = os.path.join(BASE_DIR, "input.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "output.txt")

MODEL = "$MODEL"

# --- Load or create vector store ---
if not os.path.exists(DB_DIR):
    print("📚 Indexing local files...")
    loaders = []
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        if file.endswith(".pdf"):
            loaders.append(PyPDFLoader(path))
        elif file.endswith(".txt") or file.endswith(".py") or file.endswith(".ipynb"):
            loaders.append(TextLoader(path))
    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model=MODEL)
    db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=DB_DIR)
    db.persist()
else:
    from langchain.vectorstores import Chroma
    embeddings = OllamaEmbeddings(model=MODEL)
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

retriever = db.as_retriever(search_kwargs={"k": 3})
llm = Ollama(model=MODEL)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# --- Read query ---
if not os.path.exists(INPUT_FILE):
    print(f"⚠️ Input file not found: {INPUT_FILE}")
    exit(1)

with open(INPUT_FILE, "r") as f:
    query = f.read().strip()

# --- Run AI ---
print(f"🧩 Processing: {query[:80]}...")
answer = qa.run(query)

# --- Save answer ---
with open(OUTPUT_FILE, "w") as f:
    f.write(answer.strip())

print(f"✅ Answer saved to {OUTPUT_FILE}")
EOF

# ---- Step 6: Shell script to query ----
cat > ask_ai.sh <<'EOF'
#!/bin/bash
cd ~/offline_ai/scripts
python3 ai_query.py
EOF
chmod +x ask_ai.sh

echo "🎯 Setup complete!"
echo ""
echo "To use:"
echo "1️⃣ Put your PDFs, codes, notes into ~/offline_ai/data"
echo "2️⃣ Write your question in ~/offline_ai/input.txt"
echo "3️⃣ Run: ~/offline_ai/ask_ai.sh"
echo "4️⃣ View output: ~/offline_ai/output.txt"
echo ""
echo "💡 Example:"
echo "echo 'Write a Python implementation of A* search algorithm with explanation' > ~/offline_ai/input.txt"
echo "~/offline_ai/ask_ai.sh"
echo "cat ~/offline_ai/output.txt"
echo ""
echo "⚙️ Model used: $MODEL"

