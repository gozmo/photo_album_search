import trainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_name")
args = parser.parse_args()

trainer.train(args.model_name)
