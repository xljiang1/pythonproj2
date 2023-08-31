
import genstudiopy
import glob
import sys
from genstudiopy.langchain_plugin.embeddings import GenStudioOpenAIEmbeddings
from genstudiopy.langchain_plugin.llms import GenStudioChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import os

ROOT_DIR = os.path.dirname(__file__)
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
PERSIST_DIRECTORY = os.path.join(ROOT_DIR, 'db')
llm = GenStudioChatOpenAI(temperature=0)
MIN_DOCS = 3


def main():
    result = process_request()
    print(result)
    return result
    # print(make_query(retrievalQA_chain, "what is upstreamTime field in service mesh"))
    # print(make_query(retrievalQA_chain, "What is Envoy"))
    # print(make_query(retrievalQA_chain, "explain gRPC support"))
    # print(make_query(retrievalQA_chain, "What language can Envoy work with"))
    # print(make_query(retrievalQA_chain, "What kind of proxy Envoy is"))
    # print(make_query(retrievalQA_chain, "What is Envoy's goal"))
    # print(make_query(retrievalQA_chain, "I am getting 504 error while calling service over api gateway, how can I fix it"))
    # print(make_query(retrievalQA_chain, "how can I fix gateway timeout error"))
    # print(make_query(retrievalQA_chain, "how to read service mesh log fields"))


def process_request():
    embeddings = GenStudioOpenAIEmbeddings()

    if len(sys.argv) < 3:
        return "enough arguments to script are not passed. Possible options - python3 main.py embeddings \"<document-directory-path>\" or python3 main.py query \"<query-string>\""
    if sys.argv[1] == "embeddings":
        # creates embeddings for any changes in the documents
        result = create_embeddings(embeddings, sys.argv[2])
        return result
    elif sys.argv[1] == "query":
        # have the retrieval created
        retrievalQA_chain = create_retrival_chain(embeddings)
        result = make_query(retrievalQA_chain, sys.argv[2])
        return result
    else:
        return "execution of script requires one of these [\"embeddings\", \"query\"] as first argument"


def create_embeddings(embeddings, documentDirectoryPath):
    documents = get_text_file_documents(documentDirectoryPath)

    texts = split_documents(documents)

    vectordb = Chroma(embedding_function=embeddings, persist_directory=PERSIST_DIRECTORY)
    for i in range(len(texts)):
        vectordb.add_texts(texts=texts[i:i + 1], metadatas=[{"source": f"{i}-pl"}])

    return "vector db updated"

def create_retrival_chain(embeddings):
    vectordb = Chroma(embedding_function=embeddings, persist_directory=PERSIST_DIRECTORY)

    retrievalQA_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                                    retriever=vectordb.as_retriever(search_kwargs={"k": MIN_DOCS}))
    return retrievalQA_chain


# retrieval query
def make_query(chain, question):
    return chain.run({"query": question})


# list filenames from document directory
def get_all_text_files(directory_file_format):
    source_files = glob.glob(directory_file_format)
    return source_files


# load provided document file
def load_text_file(filename):
    loader = TextLoader(filename)
    document = loader.load()
    return document


#
def get_text_file_documents(directory_file_format):
    text_filenames = get_all_text_files(directory_file_format)
    text_file_documents = []
    for filename in text_filenames:
        text_file_documents.append(*load_text_file(filename))
    return text_file_documents


def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts_split = text_splitter.split_documents(documents)
    texts = [i.page_content for i in texts_split]
    return texts


if __name__ == '__main__':
    main()
