from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone
import tempfile

from dotenv import load_dotenv, find_dotenv
import streamlit as st 

import os 

# Load Private Data 
def load_document(file):
    name, extension = os.path.splitext(file)

    if extension == ".pdf":
        from langchain.document_loaders import PyPDFLoader
        print(f"Loading PDF file ......{file}")
        loader = PyPDFLoader(file)
        print(f"Done........")

    elif extension == ".docx":
        from langchain.document_loaders import Docx2txtLoader
        print(f"Loading docx file.......{file}")
        loader = Docx2txtLoader(file)
        print(f"Done.......")

    elif extension == ".txt":
        print(f"loading txt file.......{file}")
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file, encoding='utf-8')
        print(f"Done.......")

    else:
        print("Document format is not supported!")
        return None 

    data = loader.load()
    return data


# Load public document
def load_wikipedia_document(query, lang='en', load_max=2):
    from langchain_community.document_loaders import WikipediaLoader
    docs = WikipediaLoader(query=query, load_max_docs=load_max).load()
    return docs

# splitting the dataset into chunk 
def chunk_data(data, file_name, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=20)
    chunks = text_splitter.split_documents(data)
    return chunks

# Embedding Cost 
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    embedding_cost = total_tokens / 1000 * 0.0004
    return total_tokens, embedding_cost

# $if you have created new index and you are storing the document in to the vector then only use this function 
def embedding_and_storing(index_name, chunks):

    from langchain_pinecone import PineconeVectorStore
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    print("only applicable if you have created a new Index....")
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    print("Documents successfully uploaded to pinecone!!")

    return vectorstore


def ask_and_get_answer(vectorstore, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
    answer = chain.invoke(q)
    return answer

def summarizing(docs):

    from langchain import PromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains.summarize import load_summarize_chain

    max_prompt="Write a short and concise summary of the following " \
    "Text: {text}" \
    "consise summary: "

    combine_prompt="Write a concise summary of the following text that covers the key points. " \
    "Add a title to the summary" \
    "Start your summary with an `Introduction Paragraph` that gives an overview of" \
    "the topic followed by `BULLET POINTS` if possible and the summary with " \
    "CONCLUSION PHRASE:" \
    "Text: {text}" 

    combine_prompt_template = PromptTemplate(
        template=combine_prompt, input_variables=['text']
    )   

    max_prompt_template = PromptTemplate(
        input_variables=["text"],
        template=max_prompt
    )


    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1)

    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=max_prompt_template,
        combine_prompt=combine_prompt_template,
        verbose=False
    )

    output = chain.invoke(docs)

    return output['output_text']

def ask_with_memory(vector_store, question, k=3, chat_history=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash' , temperature=1)

    if "summarize" in question.lower():
        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 10})
        docs = retriever.invoke("summarize all content")
        answer = summarizing(docs)
        return answer
    
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})


    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))
    return result, chat_history

def delete_pinecone_index(index_name="all"):
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

    if index_name == "all":
        indexes = pc.list_indexes()
        print("Deleting all indexes...")
        for index in indexes:
            pc.delete_index(index.name)
            print(f"completed deleting index {index.name}")
    else:
        print(f"Deleting index {index_name}")
        pc.delete_index(index_name)
        print("Done...")

def creating_new_index(index_name):
    from pinecone import ServerlessSpec
    from langchain_pinecone import PineconeVectorStore
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    pc = Pinecone(
                    api_key=os.environ.get("PINECONE_API_KEY")
                )

    if index_name not in pc.list_indexes():
        # if we could not find the index-name in the pinecone we have to create a new one
        print(f"Creating an index name...........{index_name}")
        pc.create_index(
            index_name,
            dimension=3072,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print("Done creating Index..")

    else:
        print(f"Index {index_name} already exists.....", ends='')

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

def main():
    load_dotenv(find_dotenv(), override=True)

    index_name = "askadocument"
    st.image("../image.png")
    st.subheader('LLM Question-Answering Application.🤔')
    with st.sidebar:
        # api_key = st.text_input("Enter your gemini api key: ", type='password')
        # pine_key = st.text_input("Enter your pinecone api key: ", type="password")
        # if api_key:
        #     os.environ["GEMINI_API_KEY"] = api_key

        # if pine_key:
        #     os.environ['PINECONE_API_KEY'] = pine_key

        uploaded_file = st.file_uploader("Upload a file: ", type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('chunk_size: ', min_value=100, max_value=1536, value=256, on_change=clear_history)
        k = st.number_input("k :", min_value=1, max_value=5, value=3, on_change=clear_history)

        add_data = st.button("Add Data", on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner("Reading, chunking and embedding file .... loading"):
                
                from pathlib import Path
                save_folder = "./"
                save_path = Path(save_folder, uploaded_file.name)
                bytesdata = uploaded_file.read()
                with open(save_path, 'wb') as w:
                    w.write(bytesdata)

                data = load_document(save_path)

                chunks = chunk_data(data, index_name, chunk_size=chunk_size)
                st.write(f"chunks size: {len(chunks)}")
                total_token, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f"total token : {total_token}  \nembedding cost : ${embedding_cost}")

                # initalising the pinecone api key which i have stored on the env variables
                pc = Pinecone(
                    api_key=os.environ.get("PINECONE_API_KEY")
                )

                # delete existing index and will create a new index 
                delete_pinecone_index()

                # creating a new index 
                creating_new_index(index_name)
                index = pc.Index(index_name)

                vector_store = embedding_and_storing(index_name, chunks) 

                st.session_state.vs = vector_store 
                st.success('file uploaded, chunked and embedded successfully!!')

    q = st.text_input("Ask a question about the content of your file: ")
    chat_history = []
    if q:
        if "vs" in st.session_state:
            vector_store = st.session_state.vs
            # st.write(f"k: {k}  higher the k the more number of tokens it will generate")
            
            answer, chat_history = ask_with_memory(vector_store, q, k, chat_history)
            if "summarize" in q.lower():
                st.text_area(f'LLM answer:', value=answer)
            else:
                st.text_area(f'LLM answer:', value=answer["answer"])

            st.divider()
            if "history" not in st.session_state :
                st.session_state.history = ""
            value = f"q: {chat_history[-1][0]} \nA: {chat_history[-1][1]}"
            st.session_state.history = f"{value} \n {'-' * 100} \n {st.session_state.history}"
            h = st.session_state.history
            st.subheader("Chat History")
            st.text_area("", value=h, height=400, key="history", disabled=True)

if __name__ == "__main__":
    main()