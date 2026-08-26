"""
Data loading and preprocessing for QLoRA fine-tuning.
Supports Bitext customer support dataset and custom JSON formats.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


class CustomerSupportDataset:
    """Load customer support Q&A dataset."""
    
    def __init__(self, tokenizer_name: str = "meta-llama/Llama-2-7b", max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_bitext(self, split: str = "train") -> Dataset:
        """Load Bitext customer support dataset."""
        dataset = load_dataset(
            "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
            split=split
        )
        return self._format_to_chat(dataset)
    
    def load_json(self, json_path: str) -> Dataset:
        """Load custom JSON dataset.
        
        Expected format:
        [
            {
                "instruction": "Answer customer support question",
                "input": "How do I reset my password?",
                "output": "To reset your password..."
            }
        ]
        """
        with open(json_path) as f:
            data = json.load(f)
        
        texts = []
        for item in data:
            instruction = item.get("instruction", "")
            inp = item.get("input", "")
            output = item.get("output", "")
            
            # Chat template format
            text = f"<s>[INST] {instruction}\n{inp} [/INST] {output} </s>"
            texts.append({"text": text})
        
        dataset = Dataset.from_dict({"text": [t["text"] for t in texts]})
        return dataset
    
    def _format_to_chat(self, dataset: Dataset) -> Dataset:
        """Format dataset to chat template."""
        def format_fn(batch):
            texts = []
            for q, a in zip(batch.get("question", []), batch.get("answer", [])):
                text = f"<s>[INST] {q} [/INST] {a} </s>"
                texts.append(text)
            return {"text": texts}
        
        return dataset.map(format_fn, batched=True, remove_columns=dataset.column_names)
    
    def tokenize_batch(self, examples: Dict) -> Dict:
        """Tokenize batch for training."""
        inputs = self.tokenizer(
            examples["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs
    
    def prepare_train_dataset(self, json_path: str) -> Dataset:
        """Prepare training dataset."""
        dataset = self.load_json(json_path)
        dataset = dataset.map(self.tokenize_batch, batched=True, remove_columns=["text"])
        return dataset


if __name__ == "__main__":
    loader = CustomerSupportDataset()
    
    # Create sample data
    samples = [
        {
            "instruction": "Answer customer support question",
            "input": "How do I reset my password?",
            "output": "To reset: 1) Click Forgot Password 2) Enter email 3) Check email for link 4) Create new password"
        },
        {
            "instruction": "Answer customer support question",
            "input": "What's your refund policy?",
            "output": "30-day money-back guarantee. Contact support@example.com with order number. Refunds in 5-7 business days."
        }
    ]
    
    Path("data/train").mkdir(parents=True, exist_ok=True)
    with open("data/train/sample.json", "w") as f:
        json.dump(samples, f, indent=2)
    
    print("✓ Sample dataset created")
