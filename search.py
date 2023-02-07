class PictureSearch:
    def __init__(self):
        self.milvus_connection = Collection("image_vectors")      # Get an existing collection.

    def search(self, text_embedding):

        results = collection.search(data=[text_embedding], 
                                    anns_field="image_vectors", 
                                    param=search_params,
                                    limit=10, 
                                    expr=None,
                                    consistency_level="Strong"
                                    )
