import os
import uuid
from codectx_ai.embeddings.utils.CodeSplitter import create_chunks, create_documentation
from langchain_community.document_loaders import DirectoryLoader
from langchain_elasticsearch import ElasticsearchStore
from langchain_google_genai import embeddings

embedding = embeddings.GoogleGenerativeAIEmbeddings(model="models/embedding-001",
                                                    google_api_key=os.getenv("GOOGLE_API_KEY"))
es_api_key = os.getenv("ES_API_KEY")
es_cloud_id = os.getenv("ES_CLOUD_ID")
es_index_name = os.getenv("ES_INDEX_NAME")
es_collection_name = os.getenv("ES_COLLECTION_NAME")
vectordb = ElasticsearchStore(
    es_cloud_id=es_cloud_id,
    index_name=es_index_name,
    es_api_key=es_api_key,
    embedding=embedding,
)


def load_file(repo_path):
    extensions = ['js', 'tsx', 'ts', 'jsx', 'css']
    file_type_counts = {}
    documents_dict = {}

    for ext in extensions:
        glob_pattern = f'**/*.{ext}'
        try:
            loader = DirectoryLoader(repo_path, glob=glob_pattern, show_progress=True)
            print("Loading files with pattern: ", glob_pattern)
            loaded_documents = loader.load()
            if loaded_documents:
                file_type_counts[ext] = len(loaded_documents)
                for doc in loaded_documents:
                    file_path = doc.metadata['source']
                    relative_path = os.path.relpath(file_path, repo_path)
                    file_id = str(uuid.uuid4())
                    doc.metadata['source'] = relative_path
                    doc.metadata['file_id'] = file_id

                    documents_dict[file_id] = doc
        except Exception as e:
            print(f"Error loading files with pattern '{glob_pattern}': {e}")
            continue
    return documents_dict, file_type_counts


def index_file(split_documents):
    if split_documents:
        vectordb.from_documents(split_documents, index_name=es_index_name, embedding=embedding, es_cloud_id=es_cloud_id,
                                es_api_key=es_api_key)
    return vectordb


def initiate_indexing(path):
    document_dict = {}
    file_type_counts = {}
    for root, dirs, _ in os.walk(path):
        dd, ftc = load_file(root)
        document_dict.update(dd)
        file_type_counts.update(ftc)
    split_documents = create_chunks(document_dict)
    create_documentation(split_documents)
    vdb = index_file(split_documents)
    return vdb
