import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import time
from dotenv import load_dotenv
load_dotenv()

class MongoQuerier:
    """
    Connects to MongoDB, generates query embeddings, and performs
    vector similarity search to retrieve relevant documents.
    """

    def __init__(self, mongo_uri, db_name, collection_name, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the MongoDB client and the Sentence Transformer model.

        Args:
            mongo_uri (str): The connection string for MongoDB.
            db_name (str): The name of the MongoDB database.
            collection_name (str): The name of the MongoDB collection.
            model_name (str): The name of the Sentence Transformer model to use.
                               Must match the model used for embedding generation.
        """
        try:
            self.client = MongoClient(mongo_uri)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            # Optional: Check if the collection exists and has documents
            if self.collection.count_documents({}) == 0:
                 print(f"Warning: Collection '{collection_name}' is empty or does not exist.")
            print(f"Connected to MongoDB: {db_name}/{collection_name}")
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            raise

        try:
            print(f"Loading Sentence Transformer model: {model_name}...")
            # Ensure device matches the one used for embedding if relevant (e.g., 'cuda')
            self.model = SentenceTransformer(model_name, device='cpu')
            # Get embedding dimension for vector search index check (optional but good practice)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"Error loading Sentence Transformer model: {e}")
            raise

        self.vector_index_name = "vector_index_1" # CHANGE THIS if your index has a different name
        print(f"Ensure a MongoDB Atlas Search index named '{self.vector_index_name}' exists on the 'embedding' field.")


    def search_similar_documents(self, query_text, top_k=5, num_candidates=50):
        """
        Generates embedding for the query and performs vector search.

        Args:
            query_text (str): The user's search query.
            top_k (int): The number of top similar documents to return.
            num_candidates (int): The number of candidates to consider during the search
                                 (higher means potentially more accuracy but slower).

        Returns:
            list: A list of dictionaries, each containing 'file_path', 'text',
                  and 'score' for the top_k similar documents. Returns None on error.
        """
        if not query_text:
            print("Error: Query text cannot be empty.")
            return None

        print(f"\nSearching for documents similar to: '{query_text}'")
        start_time = time.time()

        try:
            # 1. Generate query embedding
            query_embedding = self.model.encode(query_text, convert_to_tensor=False).tolist() # Use tolist() for JSON compatibility

            # 2. Construct the MongoDB $vectorSearch aggregation pipeline
            #    Requires a vector search index on the 'embedding' field in MongoDB Atlas
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.vector_index_name, # Make sure this matches your Atlas Search index name
                        "path": "embedding",             # Field containing the vectors
                        "queryVector": query_embedding,  # The embedding of your query
                        "numCandidates": num_candidates, # Number of nearest neighbors to retrieve before limiting
                        "limit": top_k                   # Return top_k results
                    }
                },
                {
                    "$project": {
                        # "_id": 0, # <-- REMOVE THIS LINE TO KEEP THE ID
                        "embedding": 0,                  # Exclude the embedding vector itself
                        "score": { "$meta": "vectorSearchScore" } # Include the similarity score
                        # Keep other fields like file_path, text implicitly by not excluding them
                        # Or explicitly include them if default behavior changes:
                        # "file_path": 1,
                        # "text": 1,
                        # "_id": 1 # Explicitly include _id
                    }
                }
            ]

            # 3. Execute the search
            results = list(self.collection.aggregate(pipeline))
            end_time = time.time()
            print(f"Search completed in {end_time - start_time:.2f} seconds.")

            return results

        except Exception as e:
            print(f"An error occurred during vector search: {e}")
            # Specific check for common index issue
            if "index not found" in str(e).lower() or "no such index" in str(e).lower():
                 print(f"Error: Atlas Search index '{self.vector_index_name}' not found or not ready.")
                 print("Please create the vector search index in MongoDB Atlas.")
            return None

    def close_connection(self):
        """Closes the MongoDB connection."""
        if self.client:
            self.client.close()
            print("MongoDB connection closed.")

# --- Main Execution Example ---
if __name__ == "__main__":
    # Use the same connection string and DB/collection names
    MONGO_CONNECTION_STRING = os.getenv("MONGO_URI")
    DATABASE_NAME = "upsc-database"
    COLLECTION_NAME = "documents-embeddings"
    # Use the same model name that was used for embedding generation
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    # How many results to display
    NUM_RESULTS_TO_SHOW = 5

    print("Starting MongoDB query process...")
    querier = None
    try:
        querier = MongoQuerier(
            mongo_uri=MONGO_CONNECTION_STRING,
            db_name=DATABASE_NAME,
            collection_name=COLLECTION_NAME,
            model_name=EMBEDDING_MODEL
        )

        # Get user query
        user_query = input("Enter your search query: ")

        if user_query:
            # Perform the search
            search_results = querier.search_similar_documents(
                user_query,
                top_k=NUM_RESULTS_TO_SHOW
            )

            # Display results
            if search_results:
                print(f"\n--- Top {len(search_results)} Results ---")
                for i, doc in enumerate(search_results):
                    print(f"\nResult {i+1}:")
                    print(f"  Score: {doc['score']:.4f}")
                    print(f"  File Path: {doc['file_path']}")
                    # Displaying only the first 300 characters of the text for brevity
                    text_preview = doc['text'][:300].replace('\n', ' ') + "..." if len(doc['text']) > 300 else doc['text'].replace('\n', ' ')
                    print(f"  Text Preview: {text_preview}")
            elif search_results == []: # Check for empty list specifically
                 print("No matching documents found.")
            # else: An error message was already printed by search_similar_documents

    except Exception as e:
        print(f"An error occurred during the main execution: {e}")
    finally:
        if querier:
            querier.close_connection()

    print("\nScript finished.")