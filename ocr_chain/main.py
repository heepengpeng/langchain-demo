from langchain import SerpAPIWrapper, OpenAI
from langchain.agents import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

import correct_contract
import email_api
import search_owner
from contract_check import ContractCheck
from curreant_date_chain import time_func
from init_env import init_api_key
from ocr_infer_chain import ocr_agreement


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
                    description="useful for when you need to determine whether a contract is compliant"
                ),
                Tool(
                    name="Correct contract.",
                    func=correct_contract.run,
                    description="useful for when you need to correct contract"
                ),
                Tool(
                    name="Search contract owner mail",
                    func=search_owner.run,
                    description="useful for when you need to search contract owner mail"
                ),
                Tool(
                    name="Send email",
                    func=email_api.run,
                    description="useful for when you need to send email"
                ),
            ] + [time_func]
    memory = ConversationBufferMemory(memory_key="chat_history")
    no_cache_llm = OpenAI(temperature=0, cache=False)
    agent = initialize_agent(tools, no_cache_llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
                             memory=memory)
    # 从这张图片中提取出合同信息: ./data/1.jpg, 并判断合同是否合规, 然后对合同进行修正，查询合同所有者邮箱，最后把结果通过邮件发送给合同维护者
    while True:
        user_input = input("请输入你的问题，输入exit退出对话：")

        if user_input == "exit":
            print("退出对话")
            break
        print(agent.run(input=user_input))


if __name__ == '__main__':
    init_api_key()
    main()
