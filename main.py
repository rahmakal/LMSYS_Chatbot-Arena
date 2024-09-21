from transformers import BitsAndBytesConfig
from torch import bfloat16
from src.finetune import FineTuningPipeline
from src.data_loader import split_data
import torch


bnb_config = BitsAndBytesConfig(
              load_in_4bit=True,
              bnb_4bit_quant_type='nf4',
              bnb_4bit_use_double_quant=True,
              bnb_4bit_compute_dtype=bfloat16)

pipeline = FineTuningPipeline(model_name="Alibaba-NLP/gte-base-en-v1.5", 
                              num_labels=3, 
                              bnb_config=bnb_config, 
                              pooling="mean", 
                              device='cuda' if torch.cuda.is_available() else 'cpu')

data_loader, test_loader = split_data(path="/data/train.csv", 
                                      tokenizer=pipeline.model.tokenizer, 
                                      data_cols=["prompt","response_a","response_b"],
                                      target_cols=["winner_model_a","winner_model_b","winner_tie"])

pipeline.train(data_loader, test_loader)