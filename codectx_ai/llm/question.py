def ask_question(question, rag_chain):
    answer_with_sources = rag_chain.invoke(
        question,
    )
    return answer_with_sources
