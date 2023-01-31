from crawler import Crawler
from encoder import Encoder
import pudb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("root_directory")

args = parser.parse_args()



crawler = Crawler(args.root_directory, ["jpg"])

files = crawler.start()

encoder = Encoder(files)
encoder.run()

