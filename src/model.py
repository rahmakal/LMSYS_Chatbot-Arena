from transformers import AutoTokenizer,AutoModel
import torch
from torch import nn


class CustomModel(nn.Module):
    def __init__(self, model_name, num_labels, bnb_config, pooling="mean"):
        super(CustomModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name, quantization_config=bnb_config, trust_remote_code=True)
        self.hidden_size = self.base_model.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        if pooling == 'max':
            self.pooling = self.max_pooling
        elif pooling == 'mean':
            self.pooling = self.mean_pooling
        elif pooling == 'max_mean':
            self.pooling = self.max_mean_pooling
            self.hidden_size *= 2
        else:
            raise ValueError("Invalid pooling method")
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        last_hidden_state = outputs[0]
        pooled_output = self.pooling(last_hidden_state, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def max_pooling(last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        last_hidden_state[input_mask_expanded == 0] = -1e9
        max_embeddings = torch.max(last_hidden_state, 1)[0]
        return max_embeddings

    def mean_pooling(last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
    
    def max_mean_pooling(self, last_hidden_state, attention_mask):
        max_pooled_output = self.max_pooling(last_hidden_state, attention_mask)
        mean_pooled_output = self.mean_pooling(last_hidden_state, attention_mask)
        return torch.cat((max_pooled_output, mean_pooled_output), dim=1)