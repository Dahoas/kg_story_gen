import json
import os
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class ExtractiveDataset(Dataset):
	def __init__(self, tokenizer, gen_token_id, file_path='train', read_range=None, kg_max_length=150, max_length=256):

		self.input_ids = []
		self.attention_masks = []
		self.labels = []
		self.dataset = []
		count = 0

		with open(file_path, 'r') as fr:
			line_iterator = tqdm(fr, desc='processing {}'.format(file_path))
			for line in line_iterator:
				if count < 2000000:
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
					#50258 is <GEN> token
					#GEN_TOKEN = 50258
					gen_index = (input_ids == gen_token_id).nonzero(as_tuple=True)[0]
					label_mask[(gen_index+1):len(temp_encoding)] = input_ids[(gen_index+1):len(temp_encoding)]
					self.labels.append(label_mask)

	def __len__(self):
		return len(self.input_ids)

	def __getitem__(self, idx):
		return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]
