from pymilvus import connections
from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import Collection
from pymilvus import utility
from encoder import TextEncoder
from constants import Files
import json
import os
import db_utils

def connect():
    connections.connect(alias="default",
                        host='localhost', 
                        port='19530')

def get_collection(collection_name):
    collection = Collection(collection_name)      
    collection.load()
    return collection

def list_collections():
    return sorted(utility.list_collections())

def search(collection_name, search_query, ignore_vector_ids=[], n=5):
    model_name = db_utils.collection_name_to_model_name(collection_name)
    text_encoder = TextEncoder("cuda", model_name)
    text_vector = text_encoder.encode(search_query)
    text_vector = text_vector.tolist()

    collection = get_collection(collection_name)

    search_params = {"metric_type": "L2",
                     "params": {"nprobe": 2},
                     "offset": 5}

    expr = f"vector_id not in {ignore_vector_ids}"
    results = collection.search(data=text_vector, 
                                expr=expr,
                                anns_field="vector", 
                                param=search_params,
                                limit=n)
    return results

def get_database_elems(collection_name, vector_ids):
    collection = get_collection(collection_name)
    res = collection.query(expr = f"vector_id in {vector_ids}",
                           offset = 0,
                           output_fields = ["vector_id", "filepath", "vector"],
                           consistency_level="Strong")

    return res

def all_data(collection_name):
    collection = get_collection(collection_name)
    res = collection.query(expr = f"vector_id != 0 ",
                           offset = 0,
                           output_fields = ["vector_id", "filepath", "vector"],
                           consistency_level="Strong")
    return res

def drop_collection(collection_name):
    utility.drop_collection(collection_name)


def create_collection(collection_name):
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
    name=collection_name, 
    schema=schema, 
    using='default', 
    shards_num=2
    )

    index_params = {
              "metric_type":"L2",
                "index_type":"IVF_FLAT",
                  "params":{"nlist":1024}
                  }
    collection.create_index(
              field_name="vector", 
                index_params=index_params
                )

def upload_images(collection_name, filepaths, vectors):
    ids = [hash(filepath) for filepath in filepaths]
    collection = get_collection(collection_name)

    collection.insert([ids, vectors, filepaths])


def add_model_name(model_name):
    safe_collection_name = model_name.replace("/", "_").replace("-", "_")

    content = {}
    if os.path.isfile(Files.MODEL_NAMES):
        with open(Files.MODEL_NAMES, "r") as f:
            content = json.load(f)

    content[model_name] = safe_collection_name

    with open(Files.MODEL_NAMES, "w") as f:
        json.dump(content, f)


def model_name_to_collection_name(model_name):
    with open(Files.MODEL_NAMES, "r") as f:
        content = json.load(f)
    return content[model_name]

def collection_name_to_model_name(collection_name):
    with open(Files.MODEL_NAMES, "r") as f:
        content = json.load(f)
    inv_content = {v: k for k, v in content.items()}
    return inv_content[collection_name]

