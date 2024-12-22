import os
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings

NAME_MODEL = os.getenv("MODEL")
API_KEY_MODEL = os.getenv("API_KEY")

llm = OpenAI(model = NAME_MODEL, openai_api_key = API_KEY_MODEL, temperature=0)

model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

faiss_index = FAISS.load_local("faiss_index_langchain", model, allow_dangerous_deserialization = True)
retriever = faiss_index.as_retriever()

rag_system = RetrievalQA.from_llm(llm=llm, retriever=retriever)

query = "RAG چیست و چگونه عمل می‌کند؟"
answer = rag_system.run(query)

print(f"Generated Answer: {answer}")