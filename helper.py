import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline
import torch 
import textwrap 
from PyPDF2 import PdfReader 
from typing_extensions import Concatenate
from langchain.text_splitter import CharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import PromptTemplate
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env
os.environ["LANGCHAIN_API_KEY"] = str(os.getenv("LANGCHAIN_API_KEY"))
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "2.pdf_chat_router_issue_assistant"

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

# Create LLM model
model = "C:/Users/arasu/Workspace/Projects/GenAI/models/MBZUAILaMini-Flan-T5-248M/"
tokenizer = AutoTokenizer.from_pretrained(model,truncation=True)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model)
pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 500,
        do_sample = True,
        temperature = 0.3,
        top_p= 0.95
    )
llm = HuggingFacePipeline(pipeline=pipe)

# # Initialize instructor embeddings using the Hugging Face model
embeddings = HuggingFaceEmbeddings(model_name="C:/Users/arasu/Workspace/Projects/GenAI/embeddings/bge-large-en-v1.5/snapshots/d4aa6901d3a41ba39fb536a557fa166f842b0e09/")
db_path = "vector_db"

def create_vector_db():
    # Load data from pdf
    raw_text = ""
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 500,
        chunk_overlap  = 100,
        length_function = len,
    )
    for root, dirs, files in os.walk("docs"):        
        for file in files:
            if file.endswith(".pdf"):
                pdf = PdfReader("./docs/"+file)
                for i, page in enumerate(pdf.pages):
                    content = page.extract_text()
                    if content:
                        raw_text += content
    texts = text_splitter.split_text(raw_text)

    # Create a  vector database from 'text'
    vector_db = Chroma.from_texts(texts,instructor_embeddings,persist_directory=db_path)
    vector_db.persist()
    vector_db = None 

def get_qa_chain():
    # Load the vector database from the local folder
    vector_db = Chroma(persist_directory=db_path, embedding_function = instructor_embeddings)

    # Create a retriever for querying the vector database
    retriever = vector_db.as_retriever(search_kwargs={"k":3})

    template = """
    You are friendly customer care assistant trying to help user on the context provided.\
    if the question contains greetings then greet the user back. be friendly.\
    if the answer is not found in the context then reply "No Evidence Found".\
    context: {context}
    question: {question}
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )
    return qa  