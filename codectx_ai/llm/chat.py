import re

from langchain.chains.llm import LLMChain

from codectx_ai.llm.llm import init_llm, prepare_prompt
from codectx_ai.llm.question import ask_question, QuestionContext

WHITE = "\033[37m"
GREEN = "\033[32m"
RESET_COLOR = "\033[0m"
model_name = "gpt-3.5-turbo"


def chat_app(index, documents, file_type_counts, filenames):
    conversation_history = ""
    llm = init_llm()
    prompt = prepare_prompt()
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    question_context = QuestionContext(index, documents, llm_chain, model_name, conversation_history, file_type_counts, filenames)
    while True:
        try:
            user_question = input("\n" + WHITE + "Ask a question about the repository (type 'exit()' to quit): " + RESET_COLOR)
            if user_question.lower() == "exit()":
                break
            print('Thinking...')
            user_question = format_user_question(user_question)

            answer = ask_question(user_question, question_context)
            print(GREEN + '\nANSWER\n' + answer + RESET_COLOR + '\n')
            conversation_history += f"Question: {user_question}\nAnswer: {answer}\n"
        except Exception as e:
            print(f"An error occurred: {e}")
            break


def format_user_question(question):
    question = re.sub(r'\s+', ' ', question).strip()
    return question
