from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "hse-teddy-bear/xlm-roberta-russian-stock-sentiment"


@torch.no_grad()
def _load_model():
    # Используем "slow" токенайзер (use_fast=False), чтобы не пытаться
    # конвертировать tiktoken BPE и избежать ошибок конвертации.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


tokenizer, model = _load_model()


def predict_text(text: str) -> Dict:
    """
    Запускает инференс модели по входному тексту и возвращает
    словарь с предсказанным классом и вероятностями.
    """
    try:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )
        outputs = model(**encoded)
        probs: List[float] = (
            torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
        )
        pred_label = int(torch.argmax(torch.tensor(probs)).item())
        return {"label": pred_label, "probs": probs}
    except Exception as e:
        raise RuntimeError(str(e))


