import json
import os
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class GuidedEntitiesToEntityDataset(Dataset):
	def __init__(self, tokenizer, gen_token_id, file_path='train', read_range=None, kg_max_length=150, max_length=350):

		self.input_ids = []
		self.attention_masks = []
		self.labels = []
		self.dataset = []

		with open(file_path, 'r') as fr:
			line_iterator = tqdm(fr, desc='processing {}'.format(file_path))
			stories = []
			count = 0
			for line in line_iterator:
				if (count % 4) == 0:
					stories.append([])
				stories[-1].append(line)
				count = count+1
			for story in stories:
				#Goal entitiy set
				target_entities = json.loads(story[-1].strip())["entities"]
				entity_set = json.loads(story[0].strip())["entities"]
				random.shuffle(target_entities)
				random.shuffle(entity_set)
				target_entities_set = set(target_entities)
				entity_set = set(entity_set)
				story.pop(0)
				for line in story:
					graph_obj = json.loads(line.strip())
					next_entities = graph_obj['entities']
					random.shuffle(next_entities)
					next_entities_set = set(next_entities)
					#Restrict goal to one entity
					goal = next_entities[0]
					#Only giving entities for goal sentence seems suspect
					text_input = f'|<startoftext>|{entity_set} <SEP> {target_entities_set} <GEN> {goal}|<endoftext>|'
					self.dataset.append(text_input)
					temp_encoding = tokenizer(text_input)["input_ids"]
					encodings_dict = tokenizer(text_input, truncation=False,
										max_length=max_length, padding="max_length")

					input_ids = torch.tensor(encodings_dict['input_ids'])
					input_masks = torch.tensor(encodings_dict['attention_mask'])
					self.input_ids.append(input_ids)
					self.attention_masks.append(input_masks)
					label_mask = torch.zeros_like(input_ids,dtype=torch.long)-100
					#50258 is [GEN] token for gpt2 tokenizer
					#GEN_TOKEN = 50258
					gen_index = (input_ids == gen_token_id).nonzero(as_tuple=True)[0]
					label_mask[(gen_index+1):len(temp_encoding)] = input_ids[(gen_index+1):len(temp_encoding)]
					self.labels.append(label_mask)

					#Updating entity set
					entity_set = entity_set.union(next_entities_set)

	def __len__(self):
		return len(self.input_ids)

	def __getitem__(self, idx):
		return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]