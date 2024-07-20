import os
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


def init_llm():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)
    return llm


def get_template():
    return """
    Conv: {conversation_history} | Q: {question} | Docs: {numbered_documents}
    Instr:
    1. Answer based on context/docs.
    2. Focus on repo/code.
    3. Consider:
        a. Purpose/features - describe.
        b. Functions/code - provide details/samples.
        c. Setup/usage - give instructions.
    4. Unsure? Say "I am not sure".
    Answer:
    """


def prepare_prompt():
    return PromptTemplate(
        template=get_template(),
        input_variables=["conversation_history", "question", "numbered_documents"]
    )
