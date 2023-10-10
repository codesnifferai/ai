from transformers import T5ForConditionalGeneration
import torch.nn as nn
import torch

class CodeSnifferNetwork(nn.Module):
    def __init__(self, num_labels=8):
        super(CodeSnifferNetwork, self).__init__()

        self.t5_encoder = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small').encoder

        # Sequence pooling taken from https://browse.arxiv.org/pdf/2104.05704.pdf
        self.att_pool = nn.Sequential(
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
        # Add a batch dimension if it's not present
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0) if attention_mask is not None else None

        encoder_outputs = self.t5_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        attention_coefs = self.att_pool(encoder_outputs)
        pooled_output = torch.matmul(encoder_outputs.transpose(-1, -2), attention_coefs).squeeze(-2)
        print(f"Shape of pooled_output (BEFORE): {pooled_output.shape}")
        
        # Ensure pooled_output is of shape [batch_size, 512]
        pooled_output = pooled_output.transpose(-1, -2).squeeze(-1)

        probs = self.fc(pooled_output)
        print(f"Shape of encoder_outputs: {encoder_outputs.shape}")
        print(f"Shape of attention_coefs: {attention_coefs.shape}")
        print(f"Shape of pooled_output (AFTER): {pooled_output.shape}")
        return probs
