from transformers import BartTokenizer, BartForConditionalGeneration, AutoTokenizer
import json

def build_bart_tokenizer():
	tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
	tokenizer.add_tokens(['<SEP>', '<GEN>'])
	with open('relation_vocab.json', 'rb') as handle:
		relation_dict = json.load(handle)
		print('num of relations:', len(relation_dict))
	for reltype in relation_dict:
		tokenizer.add_tokens(['<{}>'.format(reltype)])
	tokenizer.save_pretrained('imagination_bart_tokenizer')

def build_gpt2_tokenizer():
	tokenizer = AutoTokenizer.from_pretrained('gpt2')
	tokenizer.add_tokens(['<SEP>', '<GEN>'])
	with open('relation_vocab.json', 'rb') as handle:
		relation_dict = json.load(handle)
		print('num of relations:', len(relation_dict))
	for reltype in relation_dict:
		tokenizer.add_tokens(['<{}>'.format(reltype)])
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.save_pretrained('imagination_gpt2_tokenizer')

def build_hybrid_tokenizer():
	tokenizer = BartTokenizer.from_pretrained('gpt2')
	tokenizer.add_tokens(['<SEP>', '<GEN>'])
	with open('relation_vocab.json', 'rb') as handle:
		relation_dict = json.load(handle)
		print('num of relations:', len(relation_dict))
	for reltype in relation_dict:
		tokenizer.add_tokens(['<{}>'.format(reltype)])
	tokenizer.save_pretrained('imagination_hybrid_tokenizer')


if __name__ == "__main__":
	#build_bart_tokenizer()
	#build_hybrid_tokenizer()
	build_gpt2_tokenizer()