import torch
import torch.nn as nn
from transformers import BertModel

class NeuralNetwork(nn.Module):
    def __init__(self, bert=BertModel.from_pretrained("bert-base-uncased"), freeze_params=True, n_labels=3, seed=42):
        super().__init__()
        self.bert = bert
        self.set_seed(seed)
        self.freeze_params = freeze_params
        if self.freeze_params:
            for param in self.bert.parameters():
                param.requires_grad = False  # Freeze BERT layers
        else:
            for param in self.bert.parameters():
                param.requires_grad = True
                
        self.n_labels = n_labels
        self.linear_layer = nn.Linear(768, 512)
        self.second_layer = nn.Linear(512, self.n_labels)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU() 
        #self.Softmax = nn.Softmax(dim=1)
     

    def set_seed(self, seed):
        """Fixer la seed pour rendre les résultats reproductibles."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # Fixer pour les convolutions déterministes
        torch.backends.cudnn.benchmark = False  # Fixer pour des environnements reproductibles

        
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state
        last_hidden_state = self.dropout(last_hidden_state)
        x = self.linear_layer(last_hidden_state[:, 0, :]) 
        x = self.relu(x)
        x = self.second_layer(x)
        #x = self.Softmax(x)
        return x

