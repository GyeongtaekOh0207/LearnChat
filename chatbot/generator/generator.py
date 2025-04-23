from typing import Iterator
from transformers import AutoModelForCausalLM, AutoTokenizer

class Generator:
    def __init__(self) -> None:
        self.model_name = "gpt2"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        self.max_length = 50
        
    def generate(self, prompt: str) -> Iterator[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        generated = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=len(input_ids[0]) + self.max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        new_text = generated_text[len(prompt):]
        for i in range(0, len(new_text), 1):
            yield new_text[i]