import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import Trainer, TrainingArguments
import wandb
import os
import random
import json
#os.environ["WANDB_DISABLED"] = 'true'

__LOG__ = True

wandb.init(entity='dahoas') if __LOG__ else None

tokenizer = BartTokenizer.from_pretrained('/home/alex/skg_multisource/imagination_gpt2_tokenizer')
model = AutoModelForCausalLM.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

class TextDataset(Dataset):
	def __init__(self, tokenizer, file_path='train', read_range=None, kg_max_length=150, max_length=256):

		self.input_ids = []
		self.attention_masks = []
		self.labels = []
		count = 0

		with open(file_path, 'r') as fr:
			line_iterator = tqdm(fr, desc='processing {}'.format(file_path))
			for line in line_iterator:
					count+=1
					graph_obj = json.loads(line.strip())

					context = graph_obj['text']
					entities = graph_obj['entities']
					random.shuffle(entities)
					goal = entities[-1]
					entities = entities[:-1]

					#Only giving entities for goal sentence seems suspect
					text_input = f'Context: {context} <SEP> Entities: {entities} <GEN> {goal}'
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

train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

save_name = 'kg-entity-gpt2'
training_args = TrainingArguments(
  output_dir=save_name,
  num_train_epochs=4,
  per_device_train_batch_size=4,
  per_device_eval_batch_size=4,
  warmup_steps=500,
  weight_decay=0.01,
  logging_dir='./logs',
  report_to='wandb',
  save_strategy='no'
)

Trainer(model=model, args=training_args, train_dataset=train_dataset,
		eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
															  'attention_mask': torch.stack([f[1] for f in data]),
															  'labels': torch.stack([f[2] for f in data])}).train()


trainer.train()

model.save_pretrained(f'{save_name}-ckpt')