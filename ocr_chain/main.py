import time

from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.cache import SQLiteCache
from langchain.callbacks.tracers import langchain
from langchain.memory import ConversationBufferMemory

from contract_check import ContractCheck
from ocr_chain.curreant_date_chain import time_func
from ocr_chain.init_env import init_api_key
from ocr_chain.ocr_infer_chain import ocr_agreement


def main():
    llm = OpenAI(temperature=0)
    search = SerpAPIWrapper()
    contract_check = ContractCheck(llm)
    tools = [

                Tool(
                    name="OCR",
                    func=ocr_agreement,
                    description="useful for when you need to answer questions about extract text from image"
                ),
                Tool(
                    name="Current Search",
                    func=search.run,
                    description="useful for when you need to answer questions about current events or the current state of the world"
                ),
                Tool(
                    name="Determine contract compliance.",
                    func=contract_check.run,
                    description="useful for when you need to determine whether the contract is compliant."
                ),
            ] + [time_func]
    langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True,
                             memory=memory)
    # 请帮我从这张图片中提取出信息: ./data/1.jpg
    while True:
        user_input = input("请输入你的问题，输入exit退出对话：")

        if user_input == "exit":
            print("退出对话")
            break
        start_time = time.time()
        print(agent.run(input=user_input))
        end_time = time.time()
        print(f"\n执行时间: {end_time - start_time}")


if __name__ == '__main__':
    init_api_key()
    main()
