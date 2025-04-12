import chromadb
import os

class ChromaDBManager:
    """
    Manages interactions with a ChromaDB collection, including loading
    documents from a directory and performing queries.
    """
    def __init__(self, collection_name="upsc_data", db_path="./chroma_db"):
        """
        Initializes the ChromaDB client and collection.

        Args:
            collection_name (str): The name of the collection to use or create.
            db_path (str, optional): The path for the persistent database.
                                     Set to None for an in-memory database.
        """
        if db_path:
            self.client = chromadb.PersistentClient(path=db_path)
            print(f"Using persistent database at: {db_path}")
        else:
            self.client = chromadb.Client()
            print("Using in-memory database.")

        self.collection_name = collection_name
        try:
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            print(f"Collection '{self.collection_name}' loaded/created. Count: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing collection '{self.collection_name}': {e}")
            # Handle error appropriately, maybe raise it or exit
            raise

    def load_documents_from_directory(self, data_directory, batch_size=100):
        """
        Scans a directory for .txt files, reads their content, and adds them
        to the ChromaDB collection using the relative file path as the ID.

        Args:
            data_directory (str): The root directory containing the text files.
            batch_size (int): The number of documents to add in each batch.
        """
        documents = []
        ids = []

        print(f"Scanning directory: {data_directory}")
        for root, dirs, files in os.walk(data_directory):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if content.strip(): # Ensure content is not empty
                                documents.append(content)
                                relative_path = os.path.relpath(file_path, data_directory)
                                ids.append(relative_path)
                            else:
                                print(f"Skipping empty file: {file_path}")
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")

        if not documents:
            print("No text files found or processed in the directory.")
            return

        print(f"Adding {len(documents)} documents to collection '{self.collection_name}'...")
        # Add in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            try:
                self.collection.add(
                    documents=batch_docs,
                    ids=batch_ids
                )
                print(f"Added batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            except Exception as e:
                print(f"Error adding batch starting at index {i}: {e}")
                # Consider logging problematic IDs/docs: print(f"Problematic IDs: {batch_ids}")
        print(f"Finished adding documents. Collection count: {self.collection.count()}")


    def query_collection(self, query_texts, n_results=2):
        """
        Performs a query against the collection.

        Args:
            query_texts (list[str]): A list of query strings.
            n_results (int): The number of results to return for each query.

        Returns:
            dict: The query results from ChromaDB, or None if the collection is empty or an error occurs.
        """
        if self.collection.count() == 0:
            print("Collection is empty. Cannot perform query.")
            return None
        try:
            results = self.collection.query(
                query_texts=query_texts,
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"Error during query: {e}")
            return None

# --- Main Execution Example ---
if __name__ == "__main__":
    # Define the directory containing the text files
    DATA_DIR = "/Users/thyag/Desktop/projects/cleaned upsc dataset"
    COLLECTION_NAME = "upsc_embeddings" # Name of the collection to use or create
    DB_PATH = "./upsc_chroma_db" # Path to store the persistent database

    # Create an instance of the manager
    try:
        db_manager = ChromaDBManager(collection_name=COLLECTION_NAME, db_path=DB_PATH)

        # Load documents from the specified directory
        # This will only add new documents if their IDs don't already exist
        db_manager.load_documents_from_directory(DATA_DIR)

        # Example query
        print("\nRunning an example query...")
        query_result = db_manager.query_collection(
            query_texts=["fundamental rights in indian constitution"],
            n_results=3
        )

        if query_result:
            print("Query results:")
            print(query_result)

    except Exception as e:
        print(f"An error occurred during the process: {e}")

    print("\nScript finished.")
