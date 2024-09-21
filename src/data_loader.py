import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split


BATCH_SIZE = 16
MAX_LEN = 2048


class LmsysDataset:
    def __init__(self, data, target=None, tokenizer=None):
        self.data = data
        self.target = target
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if(self.target is not None):
            y = torch.tensor([self.target.iloc[idx]["winner_model_a"],self.target.iloc[idx]["winner_model_b"],self.target.iloc[idx]["winner_tie"]])
        else:
            y = torch.tensor([0,0,0])

        text = self.process_text(idx)
        if(self.tokenizer is not None):
            encoding = self.tokenizer.encode_plus(text, truncation=True, padding=False, return_tensors="pt")
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            return input_ids, attention_mask, y
        else:
            return text, y
        
    def process_text(self, idx):
        prompt = self.data.iloc[idx]["prompt"]
        response_a = self.data.iloc[idx]["response_a"]
        response_b = self.data.iloc[idx]["response_b"]
        return "prompt: "+' '.join([s.strip('"') for s in prompt.strip('[]').split('","')])+" model_a: "+' '.join([s.strip('"') for s in response_a.strip('[]').split('","')])+" model_b: "+' '.join([s.strip('"') for s in response_b.strip('[]').split('","')])    
    
    def sort_by_length(self, tokenizer):
        def get_encoded_length(text):
            encoding = tokenizer.encode_plus(text, truncation=True, return_tensors="pt")
            return encoding['input_ids'].size(1)

        combined_texts = self.data.apply(lambda row: "prompt: " + ' '.join([s.strip('"') for s in row["prompt"].strip('[]').split('","')]) +
                                                        " model_a: " + ' '.join([s.strip('"') for s in row["response_a"].strip('[]').split('","')]) +
                                                        " model_b: " + ' '.join([s.strip('"') for s in row["response_b"].strip('[]').split('","')]), axis=1)
        
        lengths = combined_texts.apply(get_encoded_length)
        sorted_indices = lengths.sort_values(ascending=False).index
        self.data = self.data.loc[sorted_indices].reset_index(drop=True)
        if self.target is not None:
            self.target = self.target.loc[sorted_indices].reset_index(drop=True)

def get_data(path,data_cols,target_cols,size=50):
    df=pd.read_csv(path)
    if size is not None:
        df=df.iloc[:size,:]
    data=df[data_cols]
    target=df[target_cols]
    return data,target

def create_dataloader(data, target, tokenizer, collate_fn):
    dataset=LmsysDataset(data, target, tokenizer)
    dataset.sort_by_length(tokenizer)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    attention_mask = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    input_ids = [ids[:MAX_LEN] for ids in input_ids]
    attention_mask = [mask[:MAX_LEN] for mask in attention_mask]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.stack(labels)

    return input_ids, attention_mask, labels

def split_data(path, tokenizer, data_cols, target_cols, collate_fn=collate_fn, test_size=0.2):
    data,target = get_data(path, data_cols, target_cols)
    target['composite_label'] = target['winner_model_a'].astype(str) + \
                                target['winner_model_b'].astype(str) + \
                                target['winner_tie'].astype(str)
    data, data_test, target, target_test = train_test_split(data, target, test_size=test_size, random_state=42, stratify=target['composite_label'])
    target = target.drop(columns=['composite_label'])
    target_test = target_test.drop(columns=['composite_label'])

    data_loader = create_dataloader(data, target, tokenizer, collate_fn)
    test_loader = create_dataloader(data_test, target_test, tokenizer, collate_fn)

    return data_loader, test_loader
