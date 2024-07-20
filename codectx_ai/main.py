import tempfile
from codectx_ai.cloner.cloner import prepare_repository
from codectx_ai.embeddings.embed import initiate_indexing, validate_index
from codectx_ai.llm.chat import chat_app

list_path = "repo-list.txt"

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as repo_path:
        prepare_repository(list_path, repo_path)
        index, documents, file_type_counts, filenames = initiate_indexing(repo_path)
        validate_index(index)
        chat_app(index, documents, file_type_counts, filenames)




