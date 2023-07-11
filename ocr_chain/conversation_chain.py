from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


def init_conversation_docsearch(input_text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(input_text)

    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_texts(
        texts, embeddings, metadatas=[{"source": i} for i in range(len(texts))]
    )
    return docsearch


def get_answer_by_query(docsearch, query):
    docs = docsearch.similarity_search(query)
    template = """You are a chatbot having a conversation with a human.

    Given the following extracted parts of a long document and a question, create a final answer.

    {context}

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], template=template
    )
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
    chain = load_qa_chain(
        OpenAI(temperature=0), chain_type="stuff", memory=memory, prompt=prompt
    )
    output = chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)
    return output["output_text"]


if __name__ == '__main__':
    # init()
    with open("rental_agreement.txt", "r", encoding="utf-8") as f:
        document = f.read()
    docsearch = init_conversation_docsearch(document)
    while True:
        user_input = input("请输入内容：")

        if user_input == "exit":
            print("退出循环")
            break
        print(get_answer_by_query(docsearch, user_input))