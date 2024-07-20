from codectx_ai.embeddings.query import search_documents, format_documents


class QuestionContext:
    def __init__(self, index, documents, llm_chain, model_name, conversation_history, file_type_counts, filenames):
        self.index = index
        self.documents = documents
        self.llm_chain = llm_chain
        self.model_name = model_name
        self.conversation_history = conversation_history
        self.file_type_counts = file_type_counts
        self.filenames = filenames


def ask_question(question, context: QuestionContext):
    relevant_docs = search_documents(question, context.index, context.documents, n_results=5)

    numbered_documents = format_documents(relevant_docs)
    print("retrieved docs:")
    print(numbered_documents)
    question_context = f"The relevant documents are:\n\n{numbered_documents}"

    answer_with_sources = context.llm_chain.run(
        question=question,
        context=question_context,
        numbered_documents=numbered_documents,
        conversation_history=context.conversation_history,
    )
    return answer_with_sources
