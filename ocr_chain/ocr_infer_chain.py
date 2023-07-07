from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import (
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from pydantic import Extra


class OCRInferChain(Chain, ABC):
    """
        An example of a custom chain.
        """

    contract_path: str = ""
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
        return ["contract_path"]

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
        self.contract_path = inputs["contract_path"]
        return {self.output_key: self.read_agreement()}

    @property
    def _chain_type(self) -> str:
        return "ocr_infer"

    def read_agreement(self) -> str:
        rental_agreement_text = ""
        with open(self.contract_path, encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                rental_agreement_text += line
        return rental_agreement_text
