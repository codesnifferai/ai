from transformers import T5ForConditionalGeneration
import torch.nn as nn
import torch

class CodeSnifferNetwork(nn.Module):
    def init(self, num_labels):
        super(CodeSnifferNetwork, self).init()

        self.t5_encoder = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small').encoder

        #Sequence pooling taken from https://browse.arxiv.org/pdf/2104.05704.pdf
        self.att_pool = nn.sequential(
            nn.Linear(self.t5_encoder.config.d_model, 1),
            nn.Softmax(dim=1),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.t5_encoder.config.d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_labels),
            nn.Sigmoid()
        )


    def forward(self, input_ids, attention_mask=None):
        encoder_outputs = self.t5_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        attention_coefs = self.att_pool(encoder_outputs)
        pooled_output = torch.matmul(encoder_outputs.transpose(-1, -2), attention_coefs).squeeze(-2)
        probs = self.fc(pooled_output)
        return probs
