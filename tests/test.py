from transformers import BartTokenizer, AutoTokenizer
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy

model = AutoModelForCausalLM.from_pretrained('gpt2')
model.save_pretrained('/srv/share2/ahavrilla3/kg_story_gen/tests/here')
