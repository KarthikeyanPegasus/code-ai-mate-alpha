import os
import uuid
from rank_bm25 import BM25Okapi
from codectx_ai.embeddings.utils.clean import clean_and_tokenize
from codectx_ai.embeddings.utils.CodeSplitter import create_chunks
from langchain_community.document_loaders import DirectoryLoader


def load_file(repo_path):
    extensions = ['go','py']
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
                    if "_test.go" in file_path:
                        continue
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
    index = None
    if split_documents:
        tokenized_documents = [clean_and_tokenize(doc.page_content) for doc in split_documents]
        index = BM25Okapi(tokenized_documents)
    return index


def initiate_indexing(path):
    document_dict = {}
    file_type_counts = {}
    for root, dirs, _ in os.walk(path):
        dd, ftc = load_file(root)
        document_dict.update(dd)
        file_type_counts.update(ftc)

    split_documents = create_chunks(document_dict)
    index = index_file(split_documents)
    return index, split_documents, file_type_counts, [doc.metadata['source'] for doc in split_documents]


def validate_index(index):
    if index is None:
        print("No documents were found to index. Exiting.")
        exit()
