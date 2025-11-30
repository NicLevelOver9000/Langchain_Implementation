from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from langchain_core.runnables import RunnableMap
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load the document, reads the text from the pdf

loader = PyPDFLoader("Children's_Encyclopedia.pdf")
docs = loader.load()

print(len(docs), "pages loaded")

# Split the document into chunks on basis of about 500 characters recursively.Trying paragraph then sentences then words.Overlap of chunks of 50 characters to make semantical sense
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

chunks = splitter.split_documents(docs)

# Embedding the chunks in vectors by hugging face model as google and openAI requires billing account and credit information
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")


load_dotenv()  # <-- loads .env automatically


# Free llm model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a junior research assistant.
        Always answer ONLY using the context provided.
        You MUST answer ONLY using the context provided.
        If the answer is not in the context, then check uour general knowledge.""".strip()
    ),
    (
        "user",
        "Context:\n{context}"
    ),
    (
        "user",
        "Question:\n{question}"
    )
])

query = input("What would you like to know about? ")

# Creating vector database using FAISS
vectordb = FAISS.from_documents(chunks, embeddings)
retriever = vectordb.as_retriever()


def debug_context(inputs):
    print("\n===== CONTEXT SENT TO MODEL =====")
    print(inputs["context"])
    print("=================================\n")
    return inputs


# RAG chain implementation by passing retriver(vector DB) and query as question
rag_chain = (
    RunnableMap({"context": retriever, "question": RunnablePassthrough()})
    | debug_context
    | prompt
    | llm
)

result = rag_chain.invoke(query)
print(result)
