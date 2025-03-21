import streamlit as st
from pymilvus import connections
from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import Collection
from PIL import Image
from io_utils import save_annotations
from io_utils import load_annotations
from io_utils import list_labels
from db_utils import connect 
import db_utils
import trainer


def search():
    collections = db_utils.list_collections()
    selected_collection = st.selectbox("Collection", collections)

    search_query = st.text_input("Search Query")
    result_n = st.number_input("Result n", min_value=0, max_value=100, value=3, step=1)
    
    if st.button("Search"):
        results = db_utils.search(selected_collection, search_query, n=result_n)
        vector_ids = [res.id for res in results[0]]
        filepaths = db_utils.get_database_elems(selected_collection, vector_ids)

        for elem in filepaths:
            filepath = elem["filepath"]
            image = Image.open(filepath)
            st.image(image)



connect()    
search()
