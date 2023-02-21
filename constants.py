class Directories:
    MODEL_REPO = "models/"
    ANNOTATION = "data/annotations"
    IMAGE_CACHE = "data/image_cache"
    CONFIG_FILES = "configs"


class Files:
    MODEL_NAMES = f"{Directories.CONFIG_FILES}/model_table.json"


COLLECTION_NAME = "image_vectors"
DEFAULT_MODEL = "openai/clip-vit-base-patch32"
