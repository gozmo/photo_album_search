from pymilvus import connections
from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import Collection
from encoder import TextEncoder

def init():
    connections.connect(alias="default",
                        host='localhost', 
                        port='19530')

def get_collection():
    collection = Collection("image_vectors")      # Get an existing collection.
    collection.load()
    return collection

def get_text_encoder():
    return TextEncoder()

def search(search_query, ignore_vector_ids=[], n=5):
    text_encoder = get_text_encoder()
    text_vector = text_encoder.encode(search_query)
    text_vector = text_vector.tolist()

    collection = get_collection()

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

def get_database_elems(vector_ids):
    collection = get_collection()
    res = collection.query(expr = f"vector_id in {vector_ids}",
                           offset = 0,
                           output_fields = ["vector_id", "filepath", "vector"],
                           consistency_level="Strong")

    return res

def all_data():
    collection = get_collection()
    res = collection.query(expr = f"vector_id != 0 ",
                           offset = 0,
                           output_fields = ["vector_id", "filepath", "vector"],
                           consistency_level="Strong")
    return res

def create_collection():
    connections.connect(
      alias="default",
      host='localhost', 
      port='19530'
    )

    collection_name = "image_vectors"

    utility.drop_collection(collection_name)


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
    self.collection.create_index(
              field_name="vector", 
                index_params=index_params
                )

def upload_images(filepaths, vectors):
    ids = [hash(filepath) for filepath in filepaths]

    self.collection.insert([ids, vectors, filepaths])

