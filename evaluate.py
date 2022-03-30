import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy

from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import Trainer, TrainingArguments
import json
import random

tokenizer = AutoTokenizer.from_pretrained('/home/alex/skg_multisource/imagination_gpt2_tokenizer')
model = AutoModelForCausalLM.from_pretrained('/home/alex/kg_story_gen/kg-entity-gpt2-ckpt')
model.resize_token_embeddings(len(tokenizer))

from tqdm import tqdm

class TextDataset(Dataset):
	def __init__(self, tokenizer, file_path='train', read_range=None, kg_max_length=150, max_length=256):

		self.input_ids = []
		self.attention_masks = []
		self.labels = []
		self.dataset = []
		count = 0

		with open(file_path, 'r') as fr:
			line_iterator = tqdm(fr, desc='processing {}'.format(file_path))
			for line in line_iterator:
				if count < 5000:
					count+=1
					graph_obj = json.loads(line.strip())

					context = graph_obj['text']
					entities = graph_obj['entities']
					random.shuffle(entities)
					goal = entities[-1:]
					entities = entities[:-1]

					#Only giving entities for goal sentence seems suspect
					text_input = f'Context: {context} <SEP> Entities: {entities} <GEN> {goal}'
					self.dataset.append(text_input)
					temp_encoding = tokenizer(text_input)["input_ids"]
					encodings_dict = tokenizer(text_input, truncation=True,
										max_length=max_length, padding="max_length")

					input_ids = torch.tensor(encodings_dict['input_ids'])
					input_masks = torch.tensor(encodings_dict['attention_mask'])
					self.input_ids.append(input_ids)
					self.attention_masks.append(input_masks)
					label_mask = torch.zeros_like(input_ids,dtype=torch.long)-100
					#50262 is [GEN] token
					GEN_TOKEN = 50262
					gen_index = (input_ids == GEN_TOKEN).nonzero(as_tuple=True)[0]
					label_mask[(gen_index+1):len(temp_encoding)] = input_ids[(gen_index+1):len(temp_encoding)]
					self.labels.append(label_mask)

	def __len__(self):
		return len(self.input_ids)

	def __getitem__(self, idx):
		return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]

filepath = '/home/alex/skg_multisource/data/train.json'
dataset = TextDataset(tokenizer, file_path=filepath)
dataset = dataset.dataset[:10]

model.to('cuda')

handcrafted_dataset = ['Context: A man . <SEP> Entities: [man, run] <GEN> ',
					   'Context: Carrie got out of her car but she forgot her keys. <SEP> Entities: [Carrie, car] <GEN> ',
					   'Context: Bob went to eat lunch at the local diner. <SEP> Entities: [Bob, lunch] <GEN> ']

dataset = handcrafted_dataset

with open('out.txt', 'w') as f:
	for datapoint in dataset:
		input = datapoint.split('<GEN>')[0]
		target = datapoint.split('<GEN>')[1]
		encoded_input = tokenizer.encode_plus(input, return_tensors='pt').to('cuda')
		output = model.generate(**encoded_input)
		decoded_output = tokenizer.decode(output[0])
		f.write(f'Input: {input}\n')
		f.write(f'Decoded output: {decoded_output}\n')
		f.write(f'Target: {target}\n\n')

