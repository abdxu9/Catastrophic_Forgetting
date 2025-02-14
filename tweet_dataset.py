import torch
import transformers
from transformers import BertTokenizer
from torch.utils.data import Dataset

TOKENIZER = torch.hub.load('huggingface/pytorch-transformers','tokenizer', "bert-base-uncased")
MAX_LENGTH  = 280
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

class TweetDataset(Dataset):
    """
    
    """
    def __init__(self, encodings, tokenizer=TOKENIZER, max_length=MAX_LENGTH, device=DEVICE):
        self.encodings = encodings
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __getitem__(self, index):
        text = self.encodings[index]
        inputs = self.tokenizer(
            text=text['text'],
            padding='max_length',  # Change padding to max_length
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        target = text['label']
        target = 0 if target == 0 else 1 #if target == 1 else [0,0,1]
        
        return {
            'ids': inputs['input_ids'].squeeze(0).to(self.device),
            'token_type_ids': inputs['token_type_ids'].squeeze(0).to(self.device),
            'attention_mask': inputs['attention_mask'].squeeze(0).to(self.device),
            'target': torch.tensor(target, dtype=torch.float).to(self.device)
        }

    def __len__(self):
        return len(self.encodings)
