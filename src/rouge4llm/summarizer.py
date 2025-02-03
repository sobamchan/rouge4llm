from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
import torch

from tqdm import tqdm
from transformers import (
    Pipeline,
    pipeline,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)


@dataclass(frozen=True)
class Summarizer:
    def summarize(self, docs: list[str]) -> list[str]:
        raise NotImplementedError()


@dataclass(frozen=True)
class LLaMASummarizer:
    pipe: Pipeline
    system_instruction: str

    @classmethod
    def load(cls, model_name: str, system_instruction: str) -> "LLaMASummarizer":
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=bnb_config,
            )
        except PackageNotFoundError:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
            )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=AutoTokenizer.from_pretrained(model_name),
        )
        return cls(pipe=pipe, system_instruction=system_instruction)

    def summarize(self, docs: list[str]) -> list[str]:
        cands = []
        for doc in tqdm(docs):
            msg = [
                {"role": "system", "content": self.system_instruction},
                {"role": "user", "content": doc},
            ]
            outputs = self.pipe(
                msg, max_new_tokens=128, do_sample=False, top_p=None, temperature=None
            )
            cand = outputs[0]["generated_text"][-1]["content"]
            cands.append(cand)
        return cands
