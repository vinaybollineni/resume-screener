import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# This is where your resume files are located
RESUME_DIR = "resumes"
# This is where the vectorized database will be saved
DB_FAISS_PATH = "local_store/faiss_index"

def create_vector_store():
    documents = []
    for filename in os.listdir(RESUME_DIR):
        file_path = os.path.join(RESUME_DIR, filename)
        try:
            if filename.lower().endswith('.pdf'):
                # Use PyPDFLoader for PDF files
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif filename.lower().endswith(('.doc', '.docx')):
                # Use Docx2txtLoader for Word documents
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

    if not documents:
        print("No valid documents found to process.")
        return

    # This will use the "nomic-embed-text" model from Ollama
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Create the FAISS vector store from the documents
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save the vector store to disk
    vectorstore.save_local(DB_FAISS_PATH)
    print(f"Vector store created successfully with {len(documents)} documents.")

if __name__ == "__main__":
    create_vector_store()