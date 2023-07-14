from langchain import SerpAPIWrapper, OpenAI
from langchain.agents import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

from contract_check import ContractCheck
from init_env import init_api_key
from ocr_chain.curreant_date_chain import time_func
from ocr_chain.ocr_infer_chain import ocr_agreement


# 请帮我判断合同是否合规

def main():
    search = SerpAPIWrapper()
    cache_llm = OpenAI(temperature=0)
    contract_check = ContractCheck(cache_llm)
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
                    description="非常有用，当你需要判断合同是否合规时"
                ),
            ] + [time_func]
    memory = ConversationBufferMemory(memory_key="chat_history")
    no_cache_llm = OpenAI(temperature=0, cache=False)
    agent = initialize_agent(tools, no_cache_llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True,
                             memory=memory)
    # 请帮我从这张图片中提取出信息: ./data/1.jpg
    while True:
        user_input = input("请输入你的问题，输入exit退出对话：")

        if user_input == "exit":
            print("退出对话")
            break
        print(agent.run(input=user_input))


if __name__ == '__main__':
    init_api_key()
    main()
