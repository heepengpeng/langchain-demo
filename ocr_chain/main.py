import os

from langchain.chains import SequentialChain

from ocr_chain.conversation_chain import init_conversation_docsearch, get_answer_by_query
from ocr_chain.data_save import DataSaveChain
from ocr_chain.extract_chain import get_extract_chain
from ocr_chain.ocr_infer_chain import OCRInferChain


def init():
    with open(".env") as f:
        line = f.readlines()[0]
        os.environ['OPENAI_API_KEY'] = line.split("=")[-1].strip()


def main():
    ocr_chain = OCRInferChain(
        output_key="input"
    )
    extrac_summary_chain = get_extract_chain(model="gpt-3.5-turbo-0613")
    data_save_chain = DataSaveChain()
    over_all_chain = SequentialChain(
        chains=[ocr_chain, extrac_summary_chain, data_save_chain],
        input_variables=["contract_path"],
        output_variables=["input", "text"],
        verbose=True
    )
    document = over_all_chain("./data/1.jpg")["input"]
    print("--- 合同识别完成 ---")
    docsearch = init_conversation_docsearch(document)
    while True:
        user_input = input("请输入你的问题，输入exit退出对话：")

        if user_input == "exit":
            print("退出对话")
            break
        print(get_answer_by_query(docsearch, user_input))


if __name__ == '__main__':
    init()
    main()
