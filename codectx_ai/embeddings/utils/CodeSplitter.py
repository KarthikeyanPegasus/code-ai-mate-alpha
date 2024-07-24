import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

from codectx_ai.embeddings.utils.RateLimitter import rate_limiter
from codectx_ai.llm.llm import init_llm, prepare_splitter_prompt

seperator = RecursiveCharacterTextSplitter.get_separators_for_language(language=Language.TS)
splitter = RecursiveCharacterTextSplitter.from_language(language=Language.TS)


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


def create_documentation(split_documents):
    i = 0
    for split_doc in split_documents:
        i += 1
        documentation = documentation_from_llm(split_doc.page_content)
        split_doc.metadata['documentation'] = documentation
        print(len(split_documents), " - ", i)
        time.sleep(5)


@rate_limiter(12)
def documentation_from_llm(code):
    llm = init_llm()
    prompt = prepare_splitter_prompt()
    llm_chain = (
            {"context": lambda x: code, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return llm_chain.invoke("")
