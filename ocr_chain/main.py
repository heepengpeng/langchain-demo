from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

from ocr_chain.conversation_chain import get_document_chain
from ocr_chain.curreant_date_chain import time
from ocr_chain.data_save import DataSaveChain
from ocr_chain.extract_chain import get_extract_chain
from ocr_chain.init_env import init_api_key
from ocr_chain.ocr_infer_chain import OCRInferChain


def main():
    ocr_chain = OCRInferChain(
        output_key="input"
    )
    # document_chain = get_document_chain()
    extrac_summary_chain = get_extract_chain(model="gpt-3.5-turbo-0613")
    data_save_chain = DataSaveChain()
    over_all_chain = SequentialChain(
        chains=[ocr_chain, extrac_summary_chain, data_save_chain],
        input_variables=["contract_path"],
        output_variables=["text"],
        verbose=True
    )
    search = SerpAPIWrapper()
    tools = [
                Tool(
                    name="OCR",
                    func=ocr_chain.run,
                    description="useful for when you need to answer questions about extract text from image"
                ),
                Tool(
                    name="Current Search",
                    func=search.run,
                    description="useful for when you need to answer questions about current events or the current state of the world"
                ),
            ] + [time]

    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = OpenAI(temperature=0)
    agent_chain = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True,
                                   memory=memory)
    # 请帮我从这张图片中提取出信息: ./data/1.jpg
    while True:
        user_input = input("请输入你的问题，输入exit退出对话：")

        if user_input == "exit":
            print("退出对话")
            break
        print(agent_chain.run(input=user_input))


if __name__ == '__main__':
    init_api_key()
    main()
