import streamlit as st
import pudb
from pymilvus import connections
from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import Collection
from PIL import Image
from io_utils import save_annotations
from io_utils import load_annotations
from io_utils import list_labels
from db_utils import init

import db_utils


def search():
    search_query = st.text_input("")
    
    if st.button("Search"):
        results = db_utils.search(search_query, n=20)
        vector_ids = [res.id for res in results[0]]
        filepaths = db_utils.get_database_elems(vector_ids)

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
    
    results = db_utils.search(search_query, ignore_vector_ids=vector_ids, n=20)
    vector_ids = [res.id for res in results[0]]
    database_elems = db_utils.get_database_elems(vector_ids)

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
        if submitted:
            save_annotations(label, annotations)


def show_annotations():
    labels = list_labels()

    selected_label = st.selectbox("Labels", labels)

    annotations = load_annotations(selected_label)
    
    vector_ids = [int(vector_id) for vector_id, is_label in annotations.items() if is_label == "True"]
    database_elems = db_utils.get_database_elems(vector_ids)

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
