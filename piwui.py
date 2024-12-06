import os
import time
import hashlib
import pickle
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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
from PyQt5.QtCore import Qt, QSize, QTimer, QMetaObject, Q_ARG, QThread, pyqtSlot

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"


class AdvancedImageIndexer:
    def __init__(self, root_directory: str, cache_path: str = 'image_index_cache.pkl'):
        self.root_directory = root_directory
        self.cache_path = cache_path
        self.index = None
        self.failed_files = set()  # Track files that fail embedding generation

        self.file_paths = []
        self.file_hash_cache = {}  # Store file hashes for change detection
        self.last_indexed_time = None
        # Supported image file extensions
        self.IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')
        # Semantic concept mappings for improved matching
        self.semantic_concepts = {
            'fire': ['fire', 'flame', 'burning', 'blaze', 'inferno', 'fireball'],
            'water': ['water', 'ocean', 'sea', 'river', 'lake', 'wave', 'liquid'],
            'animal': ['animal', 'creature', 'beast', 'wildlife', 'mammal', 'predator'],
        }

        # Validate root directory
        if not os.path.isdir(root_directory):
            raise ValueError(f"Invalid directory: {root_directory}")

        # Initialize model
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Add a new attribute for tracking pending updates
        self.pending_updates = set()
        # Add a small delay timer to batch updates
        self.update_timer = None

    def compute_file_hash(self, file_path: str) -> str:
        """Compute a hash for the file to detect changes."""
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                hasher.update(f.read())
            return hasher.hexdigest()
        except Exception as e:
            logger.warning(f"Could not hash file {file_path}: {e}")
            return ""

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

    def load_cache(self):
        """Load cached index if available."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self.index = data['index']
                    self.file_paths = data['file_paths']
                    self.file_hash_cache = data.get('file_hash_cache', {})
                    self.last_indexed_time = data.get('last_indexed_time', None)
                    logger.info("Loaded cached index.")
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")

    def save_cache(self):
        """Save the current index and file paths to cache."""
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump({
                    'index': self.index,
                    'file_paths': self.file_paths,
                    'file_hash_cache': self.file_hash_cache,
                    'last_indexed_time': self.last_indexed_time,
                }, f)
                logger.info("Cache saved.")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def index_files(self, force_full_reload=False):
        """
        Index new or changed files with optional force full reload.
        Improved to handle batch updates and reduce unnecessary reindexing.
        """
        # If force_full_reload is True or no index exists, do a complete reload
        if force_full_reload or self.index is None:
            logger.info("Performing full index reload")
            self.reload_index()
            return

        all_image_files = self.find_image_files()
        new_or_updated_files = []

        # Track files that need indexing
        for file_path in all_image_files:
            current_hash = self.compute_file_hash(file_path)
            if self.file_hash_cache.get(file_path) != current_hash:
                new_or_updated_files.append(file_path)
                self.file_hash_cache[file_path] = current_hash

        if not new_or_updated_files:
            logger.info("No new or updated files detected.")
            return

        embeddings = []
        for file_path in new_or_updated_files:
            if file_path in self.failed_files:
                logger.info(f"Skipping previously failed file: {file_path}")
                continue

            if self.is_valid_image(file_path):
                embedding = self.get_image_embedding(file_path)
                if embedding is not None:
                    embeddings.append((file_path, embedding))
                else:
                    self.failed_files.add(file_path)
                    logger.error(f"Failed to generate embedding for: {file_path}")

        if embeddings:
            # If index doesn't exist, create it
            if self.index is None:
                self.index = faiss.IndexFlatL2(512)

            # Safely add new embeddings
            new_embeddings = np.array([embed[1] for embed in embeddings])
            self.index.add(new_embeddings)
            self.file_paths.extend([embed[0] for embed in embeddings])

            # Save cache and log
            self.save_cache()
            logger.info(f"Indexed {len(new_embeddings)} new or updated files.")

    def update_index_with_retry(self, max_retries=3):
        """
        Safely update index with retry mechanism to handle potential race conditions.
        """
        for attempt in range(max_retries):
            try:
                # Try to index files with some delay between attempts
                self.index_files()
                break
            except Exception as e:
                logger.warning(f"Indexing attempt {attempt + 1} failed: {e}")
                time.sleep(0.5)  # Short delay between retries

    def reload_index(self):
        """Reload the FAISS index from scratch and synchronize file paths."""
        try:
            embeddings = []
            valid_file_paths = []

            for file_path in self.find_image_files():
                if file_path in self.failed_files:
                    logger.info(f"Skipping previously failed file during reload: {file_path}")
                    continue

                embedding = self.get_image_embedding(file_path)
                if embedding is not None:
                    embeddings.append(embedding)
                    valid_file_paths.append(file_path)
                else:
                    self.failed_files.add(file_path)  # Track failures

            if embeddings:
                self.index = faiss.IndexFlatL2(512)
                self.index.add(np.array(embeddings))
                self.file_paths = valid_file_paths
                logger.info(f"FAISS index reloaded with {len(self.file_paths)} files.")
            else:
                self.index = None
                self.file_paths = []  # Clear file paths if no embeddings
                logger.warning("No embeddings found for reloading index.")
        except Exception as e:
            logger.error(f"Failed to reload FAISS index: {e}")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, float]]:
        """Semantic search across indexed images."""
        if self.index is None or len(self.file_paths) == 0:
            logger.warning("No images indexed. Search aborted.")
            return []

        try:
            # Debug log for validation
            logger.debug(f"FAISS index total entries: {self.index.ntotal}")
            logger.debug(f"File paths count: {len(self.file_paths)}")

            query_embedding = self.get_text_embedding(query)
            if query_embedding is None:
                logger.error("Failed to generate query embedding.")
                return []

            query_embedding = query_embedding.reshape(1, -1)

            distances, indices = self.index.search(query_embedding, top_k)

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if 0 <= idx < len(self.file_paths):
                    file_path = self.file_paths[idx]
                    similarity = 1 / (1 + dist)
                    results.append((file_path, dist, similarity))

            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def find_image_files(self) -> List[str]:
        """Find all image files in the root directory."""
        return [
            os.path.join(subdir, file)
            for subdir, _, files in os.walk(self.root_directory)
            for file in files
            if file.lower().endswith(self.IMAGE_EXTENSIONS)
        ]


class DirectoryMonitor(FileSystemEventHandler):
    def __init__(self, indexer, main_window):
        self.indexer = indexer
        self.main_window = main_window

    def _schedule_update(self):
        """
        Schedule a delayed update using thread-safe method
        """
        logger.info("Scheduling update via thread-safe method")
        # Use QMetaObject.invokeMethod to safely call across threads
        QMetaObject.invokeMethod(
            self.main_window,
            "schedule_index_update",
            Qt.QueuedConnection
        )

    def on_modified(self, event):
        if event.is_directory:
            return
        logger.info(f"File modified: {event.src_path}. Scheduling update.")
        self._schedule_update()

    def on_created(self, event):
        if event.is_directory:
            return
        logger.info(f"File created: {event.src_path}. Scheduling update.")
        self._schedule_update()


class AdvancedImageIndexerGUI(QMainWindow):
    def __init__(self, root_directory: str):
        super().__init__()

        # Setup indexer
        try:
            self.indexer = AdvancedImageIndexer(root_directory)
            self.indexer.load_cache()
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

        # Create timer for delayed updates
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._perform_delayed_update)

        # Add initial full index after setup
        self.index_images(force_full_reload=True)

    @pyqtSlot()
    def schedule_index_update(self):
        """
        Thread-safe method to schedule index updates
        """
        if not self.update_timer.isActive():
            self.update_timer.start(500)  # 500ms delay to batch updates

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

                desc_label = QLabel(
                    f"{os.path.basename(file_path)}\nDistance: {dist:.2f}\nSimilarity: {similarity:.2f}")
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
    def index_images(self, force_full_reload=False):
        """
        Index images and handle errors
        """
        try:
            self.indexer.index_files(force_full_reload=force_full_reload)

            if not self.indexer.index or not self.indexer.file_paths:
                QMessageBox.warning(self, "Indexing Failed", "No valid images found for indexing.")
        except Exception as e:
            QMessageBox.critical(self, "Indexing Error", str(e))

    def _perform_delayed_update(self):
        """
        Perform delayed index update in the main thread
        """
        try:
            logger.info("Performing delayed index update")
            self.indexer.update_index_with_retry()
        except Exception as e:
            logger.error(f"Update failed: {e}")

    # ... [rest of the methods remain the same]

def main():
    # Specify the directory to watch and index
    root_directory = "/Users/ericlivesay/test_pic_indexer/"

    # Initialize the application
    app = QApplication(sys.argv)

    # Create the main window
    window = AdvancedImageIndexerGUI(root_directory)
    window.show()

    # Setup directory monitoring
    observer = Observer()
    monitor = DirectoryMonitor(window.indexer, window)
    observer.schedule(monitor, root_directory, recursive=True)
    observer.start()

    try:
        # Run the application
        sys.exit(app.exec_())
    finally:
        # Ensure observer is stopped and cleaned up
        observer.stop()
        observer.join()

if __name__ == "__main__":
    main()
