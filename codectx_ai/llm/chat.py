import re
from codectx_ai.llm.llm import init_llm, prepare_chat_prompt
from codectx_ai.llm.question import ask_question
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

WHITE = "\033[37m"
GREEN = "\033[32m"
RESET_COLOR = "\033[0m"
model_name = "gpt-3.5-turbo"


def chat_app(vectordb):
    conversation_history = ""
    retriever = vectordb.as_retriever()
    llm = init_llm()
    prompt = prepare_chat_prompt()
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    while True:
        try:
            user_question = input(
                "\n" + WHITE + "Ask a question about the repository (type 'exit()' to quit): " + RESET_COLOR)
            if user_question.lower() == "exit()":
                break
            print('Thinking...')
            user_question = format_user_question(user_question)

            answer = ask_question(user_question, rag_chain)
            print(GREEN + '\nANSWER\n' + answer + RESET_COLOR + '\n')
            conversation_history += f"Question: {user_question}\nAnswer: {answer}\n"
        except Exception as e:
            print(f"An error occurred: {e}")
            break


def format_user_question(question):
    question = re.sub(r'\s+', ' ', question).strip()
    return question
