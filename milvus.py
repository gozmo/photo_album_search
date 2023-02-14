from pymilvus import connections
from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import Collection
from pymilvus import utility

def setup_milvus():
    connections.connect(
      alias="default",
      host='localhost', 
      port='19530'
    )

    # collection_name = "image_vectors"

    # utility.drop_collection(collection_name)


    # vector_id = FieldSchema(
      # name="vector_id", 
      # dtype=DataType.INT64, 
      # is_primary=True, 
    # )
    # filepath = FieldSchema(
      # name="filepath", 
      # dtype=DataType.VARCHAR, 
      # max_length=400,
    # )
    # vector = FieldSchema(
      # name="vector", 
      # dtype=DataType.FLOAT_VECTOR, 
      # dim=512,
    # )

    # schema = CollectionSchema(
      # fields=[vector_id,vector, filepath], 
      # description="Image vector search"
    # )

    # collection = Collection(
	# name=collection_name, 
	# schema=schema, 
	# using='default', 
	# shards_num=2
	# )
