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
from util import get_dataset_from_task, load_tokenizer
#os.environ["WANDB_DISABLED"] = 'true'

__LOG__ = True

wandb.init(entity='dahoas') if __LOG__ else None

def finetune(model_type, tokenizer_path, tokenizer_type, datapath, task, ckpt_path, use_ckpt, **kwargs):
	tokenizer, gen_token_id = load_tokenizer(tokenizer_path, tokenizer_type)
	if use_ckpt:
		model = AutoModelForCausalLM.from_pretrained(ckpt_path)
	else:
		model = AutoModelForCausalLM.from_pretrained(model_type)
	model.resize_token_embeddings(len(tokenizer))

	dataset = get_dataset_from_task(datapath, task, tokenizer, gen_token_id)

	train_size = int(0.9 * len(dataset))
	train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

	training_args = TrainingArguments(
	output_dir='results',
	num_train_epochs=8,
	per_device_train_batch_size=8,
	per_device_eval_batch_size=4,
	warmup_steps=500,
	weight_decay=0.01,
	logging_dir='./logs',
	report_to='wandb',
	save_strategy='no',
	logging_steps=50,
	)

	Trainer(model=model, args=training_args, train_dataset=train_dataset,
			eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
																'attention_mask': torch.stack([f[1] for f in data]),
																'labels': torch.stack([f[2] for f in data])}).train()


	model.save_pretrained(ckpt_path)

if __name__ == "__main__":
	finetune()
