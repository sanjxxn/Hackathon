import os
import warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=DeprecationWarning)
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
def build_vector_db():
    print("---1. Loading Blockchain Technology PDF Notes---")
    if not os.path.exists("docs"):
        print("Please Create a Docs folder and insert the files")
        return
    documents = []
    for file_path in Path("docs/").rglob("*.pdf"):
        try:
            reader = PdfReader(str(file_path))
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": str(file_path), "page":page_num}
                        )
                    )
        except Exception as e:
            print(f"Warning: Could not read{file_path} . Error: {e}")
    print(f"Loaded {len(documents)} pages from your notes.")
    print("\n--2. Splitting text into manageable chunks--")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks= text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks")
    print("\n--3. Generating embeddings using HuggingFace--")
    embeddings= HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    print("\n--4. Creating and Saving Local FAISS Index--")
    vector_store = FAISS.from_documents(chunks,embeddings)
    vector_store.save_local("faiss_index")
    print("Success! 'faiss_index' folder is created.")
if __name__=="__main__":
    os.makedirs("docs",exist_ok=True)
    build_vector_db()
