import streamlit as st
import pudb
from pymilvus import connections
from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import Collection
from encoder import TextEncoder
from PIL import Image
from io_utils import save_annotations
from io_utils import load_annotations
from io_utils import list_labels



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

def get_database_elems(vector_ids):
    collection = get_collection()
    res = collection.query(expr = f"vector_id in {vector_ids}",
                           offset = 0,
                           limit=len(vector_ids),
                           output_fields = ["vector_id", "filepath"],
                           consistency_level="Strong")

    return res

def __search(search_query, ignore_vector_ids=[], n=5):
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

def search():
    search_query = st.text_input("")
    
    if st.button("Search"):
        results = __search(search_query)
        vector_ids = [res.id for res in results[0]]
        filepaths = get_database_elems(vector_ids)

        for elem in filepaths:
            filepath = elem["filepath"]
            image = Image.open(filepath)
            st.image(image)

def annotate():
    label = st.text_input("label name")
    search_query = st.text_input("")

    database_elems = []
    
    annotations = load_annotations(label)
    vector_ids = [int(vector_id) for vector_id, is_label in annotations.items()]
    
    results = __search(search_query, ignore_vector_ids=vector_ids, n=3)
    vector_ids = [res.id for res in results[0]]
    database_elems = get_database_elems(vector_ids)

    with st.form("annotate"):
        annotations = {}
        for elem in database_elems:

            col1, col2 = st.columns(2)

            filepath = elem["filepath"]
            image = Image.open(filepath)
            col1.image(image)

            vector_id = elem["vector_id"]

            annotations[vector_id] = col2.radio("Is Label", ["True", "False"],  key=vector_id)

        submitted = st.form_submit_button("Submit")
        print(submitted)
        if submitted:
            print("Submitting", annotations)
            save_annotations(label, annotations)


def show_annotations():
    labels = list_labels()

    selected_label = st.selectbox("Labels", labels)

    annotations = load_annotations(selected_label)
    
    vector_ids = [int(vector_id) for vector_id, is_label in annotations.items() if is_label == "True"]
    database_elems = get_database_elems(vector_ids)

    for elem in database_elems:
        filepath = elem["filepath"]
        image = Image.open(filepath)
        st.image(image)


init()    

page = st.sidebar.radio("Page", ["search", "annotate", "show annotations"])

if page == "search":
    search()
elif page == "annotate":
    annotate()
elif page == "show annotations":
    show_annotations()


