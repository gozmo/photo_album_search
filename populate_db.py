from crawler import Crawler
from encoder import Encoder
import pudb
import argparse
from milvus import setup_milvus


parser = argparse.ArgumentParser()
parser.add_argument("root_directory")

args = parser.parse_args()


setup_milvus()
crawler = Crawler(args.root_directory, ["jpg", "CR2","cr2"])
# crawler = Crawler(args.root_directory, ["jpg"])

files = crawler.start()

encoder = Encoder(files)
encoder.run()

