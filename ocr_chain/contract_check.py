from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


def init_vector_db():
    loader = TextLoader("data/contract_rule_info.txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)
    return docsearch


class ContractCheck:

    def __init__(self, llm):
        self.docsearch = init_vector_db()
        self.llm = llm

    def run(self, query):
        docs = self.docsearch.similarity_search("租房合同规则")[0].page_content
        template = """
            请帮我判断合同是否合规
            
            你需要按照以下规则 {contract_rule}， 如果没有这些字段，则合同不合规。并返回不合规的原因。
            
            合同的内容是：
            {contract_content}

            """
        with open("./tmp/contract.txt") as f:
            contract_content = f.read()
        prompt = PromptTemplate(
            input_variables=["contract_rule", "contract_content"],
            template=template
        )
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)
        result = llm_chain.predict(contract_rule=docs, contract_content=contract_content)
        return result


if __name__ == '__main__':
    from init_env import init_api_key

    init_api_key()
    docsearch = init_vector_db()
    docs = docsearch.similarity_search("租房合同")
    print(docs[0].page_content)
