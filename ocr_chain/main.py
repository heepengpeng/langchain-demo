from dotenv import load_dotenv
from langchain.chains import SequentialChain

from extract_chain import get_extract_chain
from ocr_chain.data_save import DataSaveChain
from ocr_chain.ocr_infer_chain import OCRInferChain


def init():
    load_dotenv('.env')


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

    over_all_chain("rental_agreement.txt")


if __name__ == '__main__':
    init()
    main()
