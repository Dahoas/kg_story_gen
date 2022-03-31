import yaml
from experiments.entities_to_entity.entities_next_entity import EntitiesToEntityDataset
from experiments.extractive.extractive import ExtractiveDataset
from transformers import AutoTokenizer, BartTokenizer, GPT2Tokenizer

def load_run_config(config_path):
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)
	return config

def get_dataset_from_task(filepath, task, tokenizer, gen_token_id):
	if task == "extractive":
		data = ExtractiveDataset(tokenizer, gen_token_id, file_path=filepath)
	elif task == "entities_to_entity":
		data = EntitiesToEntityDataset(tokenizer, gen_token_id, file_path=filepath)
	else:
		raise ValueError(f"{task} is unknown task")
	return data

def load_tokenizer(tokenizer_path, tokenizer_type):
	if tokenizer_type == 'gpt2':
		tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
		gen_id = tokenizer('<GEN>')['input_ids'][0]
		return tokenizer, gen_id
	elif tokenizer_type == 'hybrid':
		tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
		gen_id = tokenizer('<GEN>')['input_ids'][1]
		return tokenizer, gen_id
	else:
		raise ValueError(f"{tokenizer_type} is unknown tokenizer type")