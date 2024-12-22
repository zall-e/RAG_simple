import faiss
import torch
import tqdm as tqdm
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer('intfloat/multilingual-e5-large').to(device)

texts = [
    "این یک متن نمونه است.",
    "Retrieval-Augmented Generation یک تکنیک قدرتمند است.",
    "مدل‌های زبانی بزرگ مانند GPT می‌توانند مفید باشند."
]

embeddings = model.encode(texts, convert_to_numpy=True)
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "my_faiss_index.index")
loaded_index = faiss.read_index("my_faiss_index.index")

query = "RAG یک روش کاربردی است."
query_embedding = model.encode([query], convert_to_numpy=True)

k = 2
distances, idx = loaded_index.search(query_embedding, k)

for i in idx[0]:
    print("Similar text:", texts[i])