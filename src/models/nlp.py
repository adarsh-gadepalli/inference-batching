import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
import time
from .base import BaseModel

class NLPModel(BaseModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def load(self):
        print(f"loading model {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print("model loaded.")

    @torch.inference_mode()
    def predict(self, texts: List[str]) -> List[Dict]:
        if not texts:
            return []
            
        # tokenize batch
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        ).to(self.device)

        # inference
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # format results
        results = []
        for prob in probs:
            score, label_idx = torch.max(prob, dim=0)
            label = self.model.config.id2label[label_idx.item()]
            results.append({"label": label, "score": score.item()})
            
        return results

