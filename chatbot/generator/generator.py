from typing import Iterator
from llama_cpp import Llama

class Generator:
    def __init__(self) -> None:
        self.llm = Llama.from_pretrained(
            repo_id="bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
            filename="DeepSeek-R1-Distill-Qwen-1.5B-IQ2_M.gguf",
            verbose=False 
        )
        self.max_length = 50
        
    def generate(self, prompt: str) -> str:
        response = self.llm(
            prompt,
            max_tokens=self.max_length,
            stream=False,
            echo=False
        )
        return response['choices'][0]['text']