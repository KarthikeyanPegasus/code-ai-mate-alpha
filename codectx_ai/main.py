import os
import tempfile

from langchain_elasticsearch import ElasticsearchStore

from codectx_ai.cloner.cloner import prepare_repository
from codectx_ai.embeddings.embed import initiate_indexing
from codectx_ai.llm.chat import chat_app
from langchain_google_genai import embeddings


list_path = "/Users/karthikarthi/Desktop/code-ai-mate-alpha/codectx_ai/repo-list.txt"
es_api_key = os.getenv("ES_API_KEY")
es_cloud_id = os.getenv("ES_CLOUD_ID")
es_index_name = os.getenv("ES_INDEX_NAME")
es_collection_name = os.getenv("ES_COLLECTION_NAME")
embedding = embeddings.GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))

vectordb = ElasticsearchStore(
    es_cloud_id=es_cloud_id,
    index_name=es_index_name,
    es_api_key=es_api_key,
    embedding=embedding,
)


def feed_chatbot():
    with tempfile.TemporaryDirectory() as repo_path:
        prepare_repository(list_path, repo_path)
        _ = initiate_indexing(repo_path)
        chat_app(vectordb)


if __name__ == "__main__":
    input = int(input("Press 0 to start the chatbot, 1 to feed the chatbot:"))
    if input == 1:
        feed_chatbot()

    chat_app(vectordb)



