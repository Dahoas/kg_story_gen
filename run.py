import argparse
from hf_trainer import finetune
from evaluate import evaluate_model
from util import load_run_config

def run(config):
	finetune(**config)
	evaluate_model(**config)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config_path", type=str)
	args = parser.parse_args()
	config = load_run_config(args.config_path)
	run(config)