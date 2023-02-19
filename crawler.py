import os
import glob

class Crawler:
    def __init__(self, root_directory, file_types):
        self.root_directory = root_directory
        self.file_types = file_types
        self.files = []

    def start(self):
        files = []
        for file_type in self.file_types:
            pattern = f"**/*.{file_type}"
            print(pattern)
            files += glob.glob(pattern,
                               recursive=True,
                               root_dir=self.root_directory)

        self.files = [f"{self.root_directory}{file}" for file in files]
        length = len(self.files)
        print(f"Crawler found {length} files")

        return self.files



