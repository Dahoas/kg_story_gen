import pandas as pd
import argparse
import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy

from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import Trainer, TrainingArguments
from util import get_dataset_from_task, load_tokenizer
from util import load_run_config

def evaluate_model(tokenizer_path, tokenizer_type, datapath, task, results_file, ckpt_path, **kwargs):
	tokenizer, gen_token_id = load_tokenizer(tokenizer_path, tokenizer_type)
	model = AutoModelForCausalLM.from_pretrained(ckpt_path)
	model.resize_token_embeddings(len(tokenizer))

	dataset = get_dataset_from_task(datapath, task, tokenizer, gen_token_id).dataset[:10]

	model.to('cuda')

	handcrafted_dataset = ['Context: A man . <SEP> Entities: [man, run] <GEN> ',
						'Context: Carrie got out of her car but she forgot her keys. <SEP> Entities: [Carrie, car] <GEN> ',
						'Context: Bob went to eat lunch at the local diner. <SEP> Entities: [Bob, lunch] <GEN> ']

	#dataset = handcrafted_dataset

	with open(results_file, 'w') as f:
		for datapoint in dataset:
			input = datapoint.split('<GEN>')[0]
			target = datapoint.split('<GEN>')[1]
			encoded_input = tokenizer.encode_plus(input+'<GEN>', return_tensors='pt').to('cuda')
			output = model.generate(**encoded_input)
			decoded_output = tokenizer.decode(output[0])
			f.write(f'Input: {input}\n')
			f.write(f'Decoded output: {decoded_output}\n')
			f.write(f'Target: {target}\n\n')

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config_path", type=str)
	args = parser.parse_args()
	config = load_run_config(args.config_path)
	evaluate_model(**config)
