import os
import sys
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
import logging
from typing import List, Tuple, Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLineEdit,
    QLabel, QWidget, QScrollArea, QGridLayout, QPushButton, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"


class AdvancedImageIndexer:
    def __init__(self, root_directory: str, cache_path: str = 'image_index_cache.pkl'):
        """
        Advanced image indexer with robust error handling and semantic matching

        Args:
            root_directory (str): Root directory to index images
            cache_path (str): Path to store index cache
        """
        self.root_directory = root_directory
        self.cache_path = cache_path
        self.index = None
        self.file_paths = []

        # Validate root directory
        if not os.path.isdir(root_directory):
            raise ValueError(f"Invalid directory: {root_directory}")

        # Image file extensions
        self.IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')

        # Initialize models
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Semantic concept mappings for improved matching
        self.semantic_concepts = {
            'fire': ['fire', 'flame', 'burning', 'blaze', 'inferno', 'fireball'],
            'water': ['water', 'ocean', 'sea', 'river', 'lake', 'wave', 'liquid'],
            'animal': ['animal', 'creature', 'beast', 'wildlife', 'mammal', 'predator'],
        }

    def find_image_files(self) -> List[str]:
        """
        Find all image files in the root directory and subdirectories

        Returns:
            List of full paths to image files
        """
        image_files = []
        for subdir, _, files in os.walk(self.root_directory):
            for file in files:
                # Check file extension
                if file.lower().endswith(self.IMAGE_EXTENSIONS):
                    full_path = os.path.join(subdir, file)
                    image_files.append(full_path)

        logger.info(f"Found {len(image_files)} image files")
        return image_files

    def is_valid_image(self, file_path: str) -> bool:
        """
        Validate if an image can be opened and processed
        """
        try:
            with Image.open(file_path) as img:
                # Check image dimensions and mode
                if img.width < 10 or img.height < 10:
                    return False

                # Convert to RGB to ensure compatibility
                img.convert("RGB")

                return True
        except Exception as e:
            logger.warning(f"Invalid image {file_path}: {e}")
            return False

    def get_image_embedding(self, file_path: str) -> Optional[np.ndarray]:
        """
        Generate image embedding using CLIP model

        Args:
            file_path (str): Path to the image file

        Returns:
            Optional numpy array of embedding
        """
        try:
            # Open and preprocess the image
            image = Image.open(file_path)
            image = image.convert("RGB")  # Ensure RGB mode

            # Preprocess image input
            inputs = self.processor(images=image, return_tensors="pt").to(device)

            # Generate image embedding
            with torch.no_grad():
                image_embeddings = self.model.get_image_features(**inputs)

            # Convert to numpy and normalize
            embedding = image_embeddings.cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)

            return embedding
        except Exception as e:
            logger.error(f"Image embedding generation failed for {file_path}: {e}")
            return None

    def get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate text embedding using CLIP model

        Args:
            text (str): Text to embed

        Returns:
            Optional numpy array of embedding
        """
        try:
            # Preprocess text input
            inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)

            # Generate text embedding
            with torch.no_grad():
                text_embeddings = self.model.get_text_features(**inputs)

            # Convert to numpy and normalize
            embedding = text_embeddings.cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)

            return embedding
        except Exception as e:
            logger.error(f"Text embedding generation failed: {e}")
            return None

    def index_files(self) -> Tuple[Optional[faiss.Index], List[str]]:
        """
        Advanced file indexing with comprehensive error handling

        Returns:
            Tuple of Faiss index and file paths
        """
        # Find image files
        all_image_files = self.find_image_files()

        if not all_image_files:
            logger.warning("No image files found in the specified directory")
            return None, []

        # Initialize Faiss index
        index = faiss.IndexFlatL2(512)
        valid_file_paths = []
        embeddings = []

        # Process image files
        for file_path in all_image_files:
            # Skip if image cannot be processed
            if not self.is_valid_image(file_path):
                logger.warning(f"Skipping invalid image: {file_path}")
                continue

            # Get image embedding
            embedding = self.get_image_embedding(file_path)

            # Validate embedding
            if embedding is not None and np.any(embedding):
                embeddings.append(embedding)
                valid_file_paths.append(file_path)

        # Check if we have any valid embeddings
        if not embeddings:
            logger.error("No valid image embeddings could be generated")
            return None, []

        # Add embeddings to index
        embeddings_array = np.stack(embeddings)
        index.add(embeddings_array)

        logger.info(f"Successfully indexed {len(valid_file_paths)} files")

        # Set the index and file paths
        self.index = index
        self.file_paths = valid_file_paths

        return index, valid_file_paths

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, float]]:
        """
        Semantic search across indexed images

        Args:
            query (str): Search query
            top_k (int): Number of top results to return

        Returns:
            List of tuples containing (file_path, distance, similarity)
        """
        # Check if index is initialized
        if self.index is None or len(self.file_paths) == 0:
            logger.warning("No images indexed")
            return []

        try:
            # Generate query embedding
            query_embedding = self.get_text_embedding(query)

            if query_embedding is None:
                logger.error("Failed to generate query embedding")
                return []

            # Expand query with semantic concepts
            for concept, synonyms in self.semantic_concepts.items():
                if concept in query.lower():
                    query += " " + " ".join(synonyms)

            # Reshape query embedding
            query_embedding = query_embedding.reshape(1, -1)

            # Perform search
            distances, indices = self.index.search(query_embedding, top_k)

            # Prepare results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                # Compute similarity (inverse of distance)
                similarity = 1 / (1 + dist)

                # Get file path
                file_path = self.file_paths[idx]

                results.append((file_path, dist, similarity))

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []


class AdvancedImageIndexerGUI(QMainWindow):
    def __init__(self, root_directory: str):
        super().__init__()

        # Setup indexer
        try:
            self.indexer = AdvancedImageIndexer(root_directory)
            self.index_images()
        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))
            sys.exit(1)

        # Setup UI
        self.setWindowTitle("Advanced Image Indexer")
        self.setGeometry(100, 100, 1200, 800)

        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Search bar
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        search_button = QPushButton("Search")
        search_button.clicked.connect(self.perform_search)
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(search_button)

        # Results display area
        self.results_area = QScrollArea()
        self.results_widget = QWidget()
        self.results_layout = QGridLayout()
        self.results_widget.setLayout(self.results_layout)
        self.results_area.setWidget(self.results_widget)
        self.results_area.setWidgetResizable(True)

        # Add search bar and results area to main layout
        main_layout.addLayout(search_layout)
        main_layout.addWidget(self.results_area)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def index_images(self):
        """
        Index images and handle errors
        """
        try:
            self.indexer.index_files()

            if not self.indexer.index or not self.indexer.file_paths:
                QMessageBox.warning(self, "Indexing Failed", "No valid images found for indexing.")
        except Exception as e:
            QMessageBox.critical(self, "Indexing Error", str(e))

    def perform_search(self):
        """
        Perform search and display results
        """
        query = self.search_input.text().strip()

        if not query:
            QMessageBox.warning(self, "Invalid Query", "Please enter a valid search query.")
            return

        try:
            results = self.indexer.search(query)

            if not results:
                QMessageBox.information(self, "No Results", "No matching images found.")
                return

            # Clear previous results
            self.clear_results()

            # Display results
            for idx, (file_path, dist, similarity) in enumerate(results):
                image_label = QLabel()
                image = QImage(file_path)
                pixmap = QPixmap.fromImage(image).scaled(200, 200, Qt.KeepAspectRatio)
                image_label.setPixmap(pixmap)

                desc_label = QLabel(f"{os.path.basename(file_path)}\nDistance: {dist:.2f}\nSimilarity: {similarity:.2f}")
                self.results_layout.addWidget(image_label, idx // 4, (idx % 4) * 2)
                self.results_layout.addWidget(desc_label, idx // 4, (idx % 4) * 2 + 1)

        except Exception as e:
            QMessageBox.critical(self, "Search Error", str(e))

    def clear_results(self):
        """
        Clear previous search results from the layout
        """
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    root_directory = "/Users/ericlivesay/test_pic_indexer/"  # Set your image directory path here
    window = AdvancedImageIndexerGUI(root_directory)
    window.show()

    sys.exit(app.exec_())

