import os
from dotenv import load_dotenv
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Get API key from environment
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ ERROR: GROQ_API_KEY not found in .env file")
    print("Please create a .env file with your Groq API key")
    exit(1)

# Initialize Groq client
client = Groq(api_key=api_key)

# Initialize embeddings (free, runs locally)
print("🔧 Loading embeddings model (this may take a moment)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def load_and_split_pdf(pdf_path):
    """Load PDF and split into chunks"""
    print(f"📄 Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"📄 Found {len(documents)} pages")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"📄 Split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks):
    """Create vector database from chunks"""
    print("🧠 Creating vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("✅ Vector store created!")
    return vector_store

def get_relevant_chunks(vector_store, question, k=4):
    """Get relevant chunks from vector store"""
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    return docs

def ask_groq(question, context):
    """Ask Groq with context from PDF"""
    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context.

Context from the PDF:
{context}

Question: {question}

Answer the question based ONLY on the context above. If the answer is not in the context, say "I don't have enough information to answer this question."
"""
    
    completion = client.chat.completions.create(
      model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    
    return completion.choices[0].message.content

def main():
    print("🤖 RAG Chatbot - Chat with your PDF!")
    print("=" * 50)
    
    # Ask user for PDF file path
    pdf_path = input("Enter path to your PDF file: ").strip()
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return
    
    # Load and process PDF
    chunks = load_and_split_pdf(pdf_path)
    vector_store = create_vector_store(chunks)
    
    # Chat loop
    print("\n✅ Ready! Ask questions about your PDF (type 'quit' to exit)")
    while True:
        question = input("\n🤔 Your question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        
        # Get relevant chunks
        print("🔍 Searching for relevant content...")
        docs = get_relevant_chunks(vector_store, question)
        
        # Combine context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Get answer from Groq
        print("💭 Generating answer...")
        answer = ask_groq(question, context)
        
        print(f"\n💡 Answer: {answer}")
        print("\n📚 Sources:")
        for i, doc in enumerate(docs[:3]):
            print(f"  - Source {i+1}: Page {doc.metadata.get('page', 'unknown')}")

if __name__ == "__main__":
    main()