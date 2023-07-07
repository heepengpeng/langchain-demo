import json
from abc import ABC
from typing import List, Dict, Any, Optional

import requests
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from pydantic import Extra


class DataSaveChain(Chain, ABC):
    """
        An example of a custom chain.
        """
    text: str = ""
    output_key: str = ""  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return ["text"]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        self.text = inputs["text"]
        return {self.output_key: self.http_request()}

    @property
    def _chain_type(self) -> str:
        return "save_data"

    def http_request(self) -> str:
        url = 'http://127.0.0.1:8000/api/save-contract/'  # 替换为你要发送请求的URL  # 替换为你要发送的数据
        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(url, headers=headers, data=json.dumps(self.text))

        # 检查响应状态码
        if response.status_code == 200:
            print('请求成功')
        else:
            print('请求失败')

        # 获取响应内容
        return response.text
