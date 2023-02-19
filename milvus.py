from pymilvus import connections
from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import Collection
from pymilvus import utility
from constants import COLLECTION_NAME

def connect():
    connections.connect(
      alias="default",
      host='localhost', 
      port='19530'
    )

def drop_collection():
    utility.drop_collection(COLLECTION_NAME)


def upload_images(filepaths, vectors):
    ids = [hash(filepath) for filepath in filepaths]

    collection.insert([ids, vectors, filepaths])



def create_collection():
    vector_id = FieldSchema(
      name="vector_id", 
      dtype=DataType.INT64, 
      is_primary=True, 
    )
    filepath = FieldSchema(
      name="filepath", 
      dtype=DataType.VARCHAR, 
      max_length=400,
    )
    vector = FieldSchema(
      name="vector", 
      dtype=DataType.FLOAT_VECTOR, 
      dim=512,
    )

    schema = CollectionSchema(
      fields=[vector_id,vector, filepath], 
      description="Image vector search"
    )

    collection = Collection(
    name=COLLECTION_NAME, 
    schema=schema, 
    using='default', 
    shards_num=2
    )
