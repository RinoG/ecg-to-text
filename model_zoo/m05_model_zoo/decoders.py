from transformers import DistilBertTokenizer, DistilBertModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModel, AutoTokenizer
import torch
from torch import nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
# model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
# print(f"mamba-130m-hf | #params: {sum(p.numel() for p in model.parameters())}")
# print(model)

# model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
# print(f"Bio_ClinicalBERT-hf | #params: {sum(p.numel() for p in model.parameters())}")
# print(model)

# model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b")
# print(f"opt-2.7b | #params: {sum(p.numel() for p in model.parameters())}")
# print(model)

# model = AutoModelForMaskedLM.from_pretrained("nlpie/compact-biobert")
# print(f"compact-biobert | #params: {sum(p.numel() for p in model.parameters())}")
# print(model)

# model = AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-large")
# print(f"BioLinkBERT-large | #params: {sum(p.numel() for p in model.parameters())}")
# print(model)


class ImageToTextDecoder(nn.Module):
    def __init__(self, encoder_output_dim=256, mamba_embed_dim=768):
        super(ImageToTextDecoder, self).__init__()
        # Assuming your encoder outputs at dimension 256, we need to project this to 768
        self.embedding_transform = nn.Linear(encoder_output_dim, mamba_embed_dim)
        # Initialize Mamba model
        self.mamba_model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")

    def forward(self, x):
        # x is expected to be [batch_size, seq_length, feature_dim] = [64, 16, 256]
        # Transform x to match Mamba embedding size [64, 16, 768]
        x = self.embedding_transform(x)
        # Ensure x is compatible with Mamba's input, which expects [seq_length, batch_size, feature_dim]
        x = x.permute(1, 0, 2)  # Now x is [16, 64, 768]
        # Generate outputs using Mamba
        outputs = self.mamba_model(inputs_embeds=x)
        return outputs

# Example usage
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
model = ImageToTextDecoder()
encoder_outputs = torch.randn(64, 16, 256)  # Dummy data representing your encoder output
outputs = model.forward(encoder_outputs)

logits = outputs.logits  # Shape: [16, 64, 50280]

# Convert logits to token IDs by selecting the maximum likelihood token at each position
_, predicted_token_ids = torch.max(logits, dim=-1)  # Shape: [16, 64]

# Decode token IDs to text
decoded_texts = []
for i in range(predicted_token_ids.shape[1]):  # Loop over batch size
    token_ids = predicted_token_ids[:, i].tolist()  # Get all token IDs for the i-th item in the batch
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    decoded_texts.append(decoded_text)

# Output the decoded texts
for text in decoded_texts:
    print(text)
print(outputs)