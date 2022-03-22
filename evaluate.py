import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy

from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import Trainer, TrainingArguments
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('ckpts/bart-kg-ckpt-50000')

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[1:] = input_ids[:-1].clone()
    shifted_input_ids[0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

from tqdm import tqdm
class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path='train', read_range=None, kg_max_length=900, max_length=1024):
        import csv
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        if read_range:
            start, end = read_range[0], read_range[1]
        else:
            start, end = 0, -1
        with open(file_path, 'r') as f:
            ## Read the file line by line
            # reader = csv.reader(f, delimiter=',')
            num = 0
            f = f.readlines()
            self.input_ids = []
            self.attention_masks = []
            self.decoder_input_ids = []
            self.labels = []
            for line in tqdm(f[start:end]):
                #Filter out KGs greater than some complexity
                if len(line) > kg_max_length // 2:
                  continue
                input_output = line.split('*')
                assert(len(input_output) == 2)
                input = input_output[0]
                output = input_output[1]
                input_encodings = tokenizer(input, pad_to_max_length=True, max_length=max_length)
                target_encodings = tokenizer(output, pad_to_max_length=True, max_length=max_length)

                labels = torch.tensor(target_encodings['input_ids'])
                start_token_id = 0
                decoder_input_ids = shift_tokens_right(labels, model.config.pad_token_id, start_token_id)
                labels[labels[:] == model.config.pad_token_id] = -100


                self.input_ids.append(torch.tensor(input_encodings['input_ids']))
                self.attention_masks.append(torch.tensor(input_encodings['attention_mask']))
                self.decoder_input_ids.append(torch.tensor(decoder_input_ids))
                self.labels.append(torch.tensor(labels))
                #print(input_encodings['input_ids'])
                #print(self.input_ids[-1])
                #assert(False)



    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return self.input_ids[item], self.attention_masks[item], self.decoder_input_ids[item], self.labels[item]

filepath = 'data/data_kg_diff_no_rel.txt'
dataset = []
with open(filepath, 'r') as f:
  dataset = f.readlines()
import random
random.shuffle(dataset)

model.to('cuda')

with open('out.txt', 'w') as f:
	for datapoint in dataset[:10]:
		input = datapoint.split('*')[0]
		target = datapoint.split('*')[1]
		encoded_input = tokenizer.encode_plus(input, return_tensors='pt').to('cuda')
		output = model.generate(**encoded_input)
		decoded_output = tokenizer.decode(output[0])
		f.write(f'Input: {input}\n')
		f.write(f'Decoded output: {decoded_output}\n')
		f.write(f'Target: {target}\n\n')

