import trainer
import argparse
from pymilvus import connections

connections.connect(
  alias="default",
  host='localhost', 
  port='19530')

parser = argparse.ArgumentParser()
parser.add_argument("label")

args = parser.parse_args()


trainer.train(args.label)
trainer.classify(args.label)
