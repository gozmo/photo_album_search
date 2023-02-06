import streamlit as st
import pudb
from pymilvus import connections
from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import Collection
from encoder import TextEncoder
from PIL import Image


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

def get_image_paths(vector_ids):
    collection = get_collection()
    res = collection.query(expr = f"vector_id in {vector_ids}",
                           offset = 0,
                           limit = 10, 
                           output_fields = ["vector_id", "filepath"],
                           consistency_level="Strong")

    return res

def search():
    search_query = st.text_input("")
    
    if st.button("Search"):
        text_encoder = get_text_encoder()
        text_vector = text_encoder.encode(search_query)
        text_vector = text_vector.tolist()

        collection = get_collection()

        search_params = {"metric_type": "L2", "params": {"nprobe": 2}, "offset": 5}
        results = collection.search(data=text_vector, 
                                    anns_field="vector", 
                                    param=search_params,
                                    limit=10, 
                                    expr=None,
                                    consistency_level="Strong")
        vector_ids = [res.id for res in results[0]]
        filepaths = get_image_paths(vector_ids)

        for elem in filepaths:
            filepath = elem["filepath"]
            image = Image.open(filepath)
            st.image(image)

init()    
search()
