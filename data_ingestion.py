import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import time
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

class MongoEmbedder:
    """
    Reads text files from a directory, generates embeddings using
    SentenceTransformers, and stores the text, file path, and embedding
    in MongoDB.
    """

    def __init__(self, mongo_uri, db_name, collection_name, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the MongoDB client and the Sentence Transformer model.

        Args:
            mongo_uri (str): The connection string for MongoDB.
            db_name (str): The name of the MongoDB database.
            collection_name (str): The name of the MongoDB collection.
            model_name (str): The name of the Sentence Transformer model to use.
        """
        try:
            self.client = MongoClient(mongo_uri)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            print(f"Connected to MongoDB: {db_name}/{collection_name}")
            # Create an index on file_path to prevent duplicates and speed up checks
            # Use unique=True if you want to ensure each file is added only once
            self.collection.create_index("file_path", unique=True)
            print("Index on 'file_path' created/ensured.")
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            raise

        try:
            print(f"Loading Sentence Transformer model: {model_name}...")
            # You can specify device='cuda' if you have a GPU and PyTorch installed
            self.model = SentenceTransformer(model_name, device='cpu')
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading Sentence Transformer model: {e}")
            raise

    def generate_and_store_embeddings(self, data_directory, batch_size=50):
        """
        Scans subdirectories, generates embeddings for .txt files, and stores them,
        reporting progress per subdirectory.

        Args:
            data_directory (str): The root directory containing subdirectories of text files.
            batch_size (int): The number of documents to process and insert in each batch.
        """
        total_processed_files = 0
        total_skipped_files = 0
        total_inserted_count = 0
        overall_start_time = time.time()

        # Get top-level directories
        try:
            subdirectories = [d for d in os.listdir(data_directory)
                              if os.path.isdir(os.path.join(data_directory, d))]
            if not subdirectories:
                 print(f"No subdirectories found in {data_directory}. Exiting.")
                 return
        except FileNotFoundError:
            print(f"Error: Data directory not found: {data_directory}")
            return
        except Exception as e:
            print(f"Error listing directories in {data_directory}: {e}")
            return

        print(f"Found {len(subdirectories)} subdirectories to process.")

        for sub_dir in subdirectories:
            sub_dir_path = os.path.join(data_directory, sub_dir)
            print(f"\nProcessing directory: {sub_dir}...")
            dir_start_time = time.time()
            dir_processed_files = 0
            dir_skipped_files = 0
            dir_inserted_count = 0

            # Collect all file paths for the current subdirectory first
            all_files_in_subdir = []
            for root, _, files in os.walk(sub_dir_path):
                 for file in files:
                     if file.endswith(".txt"):
                         all_files_in_subdir.append(os.path.join(root, file))

            if not all_files_in_subdir:
                print(f"No .txt files found in {sub_dir}. Skipping.")
                continue

            texts_batch = []
            paths_batch = []
            original_data_batch = []

            # Use tqdm for progress within the current subdirectory
            for file_path in tqdm(all_files_in_subdir, desc=f"Files in {sub_dir}", unit="file"):
                relative_path = os.path.relpath(file_path, data_directory)
                # Optional: Check if the document already exists based on file_path
                if self.collection.find_one({"file_path": relative_path}):
                    dir_skipped_files += 1
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():
                            texts_batch.append(content)
                            paths_batch.append(relative_path)
                            original_data_batch.append({"file_path": relative_path, "text": content})
                        else:
                            # print(f"Skipping empty file: {file_path}") # Less verbose
                            dir_skipped_files += 1
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    dir_skipped_files += 1

                # Process and insert when the batch is full
                if len(texts_batch) >= batch_size:
                    inserted = self._process_batch(texts_batch, paths_batch, original_data_batch)
                    dir_inserted_count += inserted
                    dir_processed_files += len(texts_batch) # Count processed before clearing
                    texts_batch, paths_batch, original_data_batch = [], [], [] # Reset batch

            # Process any remaining documents in the last batch for the current directory
            if texts_batch:
                 inserted = self._process_batch(texts_batch, paths_batch, original_data_batch)
                 dir_inserted_count += inserted
                 dir_processed_files += len(texts_batch)

            dir_end_time = time.time()
            print(f"Directory '{sub_dir}' processing complete.")
            print(f"  - Processed: {dir_processed_files}, Skipped: {dir_skipped_files}, Newly Inserted: {dir_inserted_count}")
            print(f"  - Time taken: {dir_end_time - dir_start_time:.2f} seconds")

            # Update overall totals
            total_processed_files += dir_processed_files
            total_skipped_files += dir_skipped_files
            total_inserted_count += dir_inserted_count


        overall_end_time = time.time()
        print("\n--- Overall Embedding Generation Complete ---")
        print(f"Total directories processed: {len(subdirectories)}")
        print(f"Total files processed: {total_processed_files}")
        print(f"Total files skipped (empty, error, or existing): {total_skipped_files}")
        print(f"Total new documents inserted: {total_inserted_count}")
        print(f"Total time taken: {overall_end_time - overall_start_time:.2f} seconds")


    def _process_batch(self, texts_batch, paths_batch, original_data_batch):
        """Helper function to embed and insert a batch."""
        if not texts_batch:
            return 0

        # print(f"Generating embeddings for batch of {len(texts_batch)} documents...") # Less verbose
        try:
            embeddings = self.model.encode(texts_batch, show_progress_bar=False) # Keep progress bar at directory level
            documents_to_insert = []
            for i, embedding in enumerate(embeddings):
                doc = original_data_batch[i]
                doc["embedding"] = embedding.tolist() # Store embedding as a list
                documents_to_insert.append(doc)

            if documents_to_insert:
                # print(f"Inserting batch of {len(documents_to_insert)} documents into MongoDB...") # Less verbose
                result = self.collection.insert_many(documents_to_insert, ordered=False) # ordered=False might be faster if errors are okay
                # print(f"Successfully inserted {len(result.inserted_ids)} documents.") # Less verbose
                return len(result.inserted_ids)
            return 0
        except Exception as e:
            print(f"\nError processing or inserting batch: {e}")
            # Log problematic paths if needed:
            # print(f"Problematic paths in failed batch: {paths_batch}")
            return 0


    def close_connection(self):
        """Closes the MongoDB connection."""
        if self.client:
            self.client.close()
            print("MongoDB connection closed.")

# --- Main Execution Example ---
if __name__ == "__main__":
    # --- Configuration ---
    MONGO_CONNECTION_STRING = os.getenv("MONGO_URI")
    DATABASE_NAME = "upsc-database"
    COLLECTION_NAME = "documents-embeddings"
    DATA_DIR = "/Users/thyag/Desktop/projects/cleaned upsc dataset"
    # Choose a model: https://www.sbert.net/docs/pretrained_models.html
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2' # Good balance of speed and quality
    # EMBEDDING_MODEL = 'all-mpnet-base-v2' # Slower, potentially higher quality
    BATCH_SIZE = 64 # Adjust based on your RAM and CPU/GPU capability

    print("Starting MongoDB embedding process...")
    embedder = None # Initialize embedder to None
    try:
        embedder = MongoEmbedder(
            mongo_uri=MONGO_CONNECTION_STRING,
            db_name=DATABASE_NAME,
            collection_name=COLLECTION_NAME,
            model_name=EMBEDDING_MODEL
        )

        embedder.generate_and_store_embeddings(DATA_DIR, batch_size=BATCH_SIZE)

    except Exception as e:
        print(f"An error occurred during the main execution: {e}")
    finally:
        if embedder:
            embedder.close_connection()

    print("\nScript finished.")