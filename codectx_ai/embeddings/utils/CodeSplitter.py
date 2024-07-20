from langchain import text_splitter
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
seperator = RecursiveCharacterTextSplitter.get_separators_for_language(language=Language.GO)
splitter = RecursiveCharacterTextSplitter.from_language(language=Language.GO)


def create_chunks(documents):
    split_documents = []
    print(seperator)
    for file_id, original_doc in documents.items():
        print(original_doc.metadata['source'])
        chunks = splitter.create_documents([original_doc.page_content])
        for chunk in chunks:
            chunk.metadata['file_id'] = original_doc.metadata['file_id']
            chunk.metadata['source'] = original_doc.metadata['source']
        split_documents.extend(chunks)
    return split_documents
