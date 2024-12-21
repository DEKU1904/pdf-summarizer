#importing libraries
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from pypdf import PdfReader


def process_text(text):
    #splitting the text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    #creating embeddings

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    #creating the FAISS index from the chunks usig the embeddings
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    return knowledge_base



def summarizer(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""


    for page in pdf_reader.pages:
        text += page.extract_text()



    knowledge_base = process_text(text)

    #showing the prompt for the user
   
    query = ("Summarize the content in the pdf approximately in 4 to 5 sentences.")

    if query:
        docs = knowledge_base.similarity_search(query)

        llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.8)

        chain = load_qa_chain(llm, chain_type="stuff")


        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            print(cb)
        return response
