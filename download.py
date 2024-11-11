#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import huggingface_hub
from transformers import BertModel, BertTokenizer

model_dir = "./models"
Path(model_dir).mkdir(parents=True, exist_ok=True)
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")


print("Downloading CT-CLIP model from huggingface...", file=sys.stderr)
actual_model_path = huggingface_hub.hf_hub_download(
    repo_id="ibrahimhamamci/CT-RATE",
    repo_type="dataset",
    filename="models_deprecated/CT_CLIP_zeroshot.pt",
    local_dir=model_dir,
    token=huggingface_token,
)
print(f"CT-CLIP model downloaded to {actual_model_path}", file=sys.stderr)
if os.path.dirname(actual_model_path) != model_dir:
    os.rename(actual_model_path, os.path.join(model_dir, "CT_CLIP_zeroshot.pt"))
    print(f"CT-CLIP model moved to {model_dir}", file=sys.stderr)


bert_model = "microsoft/BiomedVLP-CXR-BERT-specialized"
print(f"Downloading {bert_model} from huggingface", file=sys.stderr)

model = BertModel.from_pretrained(
    bert_model, trust_remote_code=True, token=huggingface_token
)
tokenizer = BertTokenizer.from_pretrained(
    bert_model, trust_remote_code=True, token=huggingface_token
)

model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print(f"{bert_model} downloaded.", file=sys.stderr)
