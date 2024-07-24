import os
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


def init_llm():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)
    return llm


prompt_sample = """
 context: {context} | Q: {question}
    In the code that i shared with:
    1. Answer based on context/docs.
    2. Focus on repo/code.
    3. Consider:
        a. Purpose/features - describe.
        b. Functions/code - provide details/samples.
        c. Setup/usage - give instructions.
    4. if you can't answer,
        a. let me know you can't.
        b. find it with your knowledge.
    Answer:
    """


def get_chat_template():
    return """
   context: {context} | Q: {question}
   In the code that i shared with:
    1) Give me Complete UI component.
    2) Tell me what are the required props for this UI component.
    3) If there is any local imports replace it with the code.
    """


def get_splitter_template():
    return """
    context: {context}
    In the code that i shared with:
    1) List the ways that this UI component can be used.
    2) what this UI component expects as input.
    documentation:
    """


def prepare_chat_prompt():
    return PromptTemplate(
        template=get_chat_template(),
        input_variables=["context", "question"]
    )


def prepare_splitter_prompt():
    return PromptTemplate(
        template=get_splitter_template(),
        input_variables=["context"]
    )
