import torch
from model.modules.sniffer import CodeSnifferNetwork
from transformers import RobertaTokenizer



class Sniffer:

    def __init__(self, model_path, num_labels=8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CodeSnifferNetwork(num_labels=num_labels)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')

    def CodeAnalysis(self, jcode):
        tokenized_jcode = self.tokenizer(jcode, return_tensors="pt")
        input_ids = tokenized_jcode.input_ids
        attention_mask = tokenized_jcode.attention_mask
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            y_pred = self.model.forward(input_ids, attention_mask)
        
        y_pred = y_pred.squeeze(0).tolist()
        y_pred = [round(num, 2) for num in y_pred]
        y_pred = dict(zip(self.model.smells, y_pred))
        return y_pred

