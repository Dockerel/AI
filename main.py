import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from webEmbeddingPipeline import WebEmbeddingPipeline
from queryProcessingPipeline import QueryProcessingPipeline

load_dotenv()

upstage_api_key = os.environ.get("UPSTAGE_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# # device = "cuda"
# # model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

device = "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2")

# # embedding pipeline
# # embedding_pipeline = WebEmbeddingPipeline(index, model)
# # embedding_pipeline.run()

# query processing pipeline
query_pipeline = QueryProcessingPipeline(index, model)
query_pipeline.run("글솝인데 졸업 요건을 알고 싶어")
