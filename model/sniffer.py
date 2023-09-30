from transformers import T5ForConditionalGeneration
import torch.nn as nn

class CodeSnifferNetwork(nn.Module):
    def __init__(self, num_labels):
        super(CodeSnifferNetwork, self).__init__()

        self.t5_encoder = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small').encoder
        self.fc = nn.Sequential(
            nn.Linear(self.t5_encoder.config.d_model, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask=None):
        encoder_outputs = self.t5_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = encoder_outputs[0][:, 0]
        probs = self.fc(pooled_output)
        return probs
