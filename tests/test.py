from transformers import BartTokenizer, AutoTokenizer

tokenizer = BartTokenizer.from_pretrained('/srv/share2/ahavrilla3/kg_story_gen/data/imagination_hybrid_tokenizer')
tok = tokenizer('<GEN>')
print(tok)
print(tokenizer.decode(tok['input_ids']))