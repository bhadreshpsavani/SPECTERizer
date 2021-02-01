from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("allenai/specter")

model = AutoModel.from_pretrained("allenai/specter")



