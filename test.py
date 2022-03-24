from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('/home/alex/skg_multisource/imagination_gpt2_tokenizer')
tok = tokenizer('<GEN>')
print(tok)
print(tokenizer.decode(tok['input_ids']))