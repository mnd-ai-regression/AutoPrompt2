from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load the tokenizer and model
model = AutoModelForMaskedLM.from_pretrained("microsoft/bitnet-b1.58-2B-4T")

# Example input
input_text = "The capital of France is [MASK]."
inputs = tokenizer(input_text, return_tensors="pt")

# Perform inference
outputs = model(**inputs)
logits = outputs.logits

# Identify the predicted token
import torch
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"Predicted token: {predicted_token}")
