import os
import re
import google.generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv
from mongo_querier import MongoQuerier
import logging
import datetime
from typing import List, Dict, Optional, Any

# --- Basic Logging Setup ---
log_directory = "logs" # Define the directory name
os.makedirs(log_directory, exist_ok=True) # Create the directory if it doesn't exist

log_filename = os.path.join(log_directory, f"rag_app_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_filename,
    filemode='w'
)

# --- Configuration Class ---
class RAGConfig:
    """Loads and stores configuration parameters."""
    def __init__(self):
        load_dotenv()
        self.mongo_uri: Optional[str] = os.getenv("MONGO_URI")
        self.gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
        self.db_name: str = "upsc-database"
        self.collection_name: str = "documents-embeddings"
        self.embedding_model: str = 'all-MiniLM-L6-v2'
        self.gemini_model_name: str = "models/gemini-2.0-flash"
        self.num_retrieved_docs: int = 10
        self.max_context_words: int = 10000
        self.max_output_tokens: int = 10000
        self.num_sub_queries: int = 5
        self.system_prompt_file: str = "adhikari-ai/prompts/system_prompt"
        self.sub_query_prompt_file: str = "adhikari-ai/prompts/sub_query_prompt"
        self.doc_id_field: str = '_id'
        self.doc_text_field: str = 'text'
        self.doc_filepath_field: str = 'file_path'
        self.gemini_temperature: float = 0.2

    def validate(self) -> bool:
        """Basic validation of essential configurations."""
        if not self.mongo_uri:
            logging.error("MONGO_URI environment variable not set.")
            print("MONGO_URI environment variable not set. Check logs.")
            return False
        if not self.gemini_api_key:
            logging.error("GEMINI_API_KEY environment variable not set.")
            print("GEMINI_API_KEY environment variable not set. Check logs.")
            return False
        return True

# --- Prompt Loader ---
class PromptLoader:
    """Loads prompts from text files."""
    @staticmethod
    def load(filepath: str) -> Optional[str]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logging.error(f"Prompt file not found at {filepath}")
            return None
        except Exception as e:
            logging.error(f"Error reading prompt file {filepath}: {e}")
            return None

# --- MongoDB Retriever ---
class MongoRetriever:
    """Handles connection and document retrieval from MongoDB."""
    def __init__(self, config: RAGConfig):
        self.config = config
        self._querier = MongoQuerier(
            mongo_uri=config.mongo_uri,
            db_name=config.db_name,
            collection_name=config.collection_name,
            model_name=config.embedding_model
        )
        logging.info("MongoDB Querier initialized.")

    def rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-ranks search results based on specific aspects of the user query."""
        # Extract key terms from the query
        query_lower = query.lower()
        
        # Check for specific question aspects
        has_definition = any(term in query_lower for term in ["who", "what is", "define", "meaning"])
        has_criticism = any(term in query_lower for term in ["criticism", "shortcoming", "weakness", "problem", "issue", "challenge", "negative"])
        has_advantage = any(term in query_lower for term in ["advantage", "strength", "benefit", "positive", "success"])
        
        # Define scoring function based on query aspects
        def relevance_score(doc):
            text = doc.get(self.config.doc_text_field, "").lower()
            score = 0
            
            # Default similarity score component
            score += doc.get("score", 0.5) * 10  # Normalize the MongoDB score
            
            # Boost documents matching specific aspects of the query
            if has_definition and any(term in text for term in ["define", "definition", "refer to", "meaning", "concept of"]):
                score += 3
            if has_criticism and any(term in text for term in ["criticism", "shortcoming", "weakness", "problem", "issue", "challenge", "negative", "failed", "failure"]):
                score += 3
            if has_advantage and any(term in text for term in ["advantage", "strength", "benefit", "positive", "success"]):
                score += 3
                
            return score
        
        # Sort documents based on the new score
        return sorted(results, key=relevance_score, reverse=True)

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        try:
            results = self._querier.search_similar_documents(query, top_k=top_k)
            return self.rerank_results(query, results)
        except Exception as e:
            logging.error(f"Error retrieving documents for query '{query}': {e}")
            return []

    def close(self):
        logging.info("Closing MongoDB connection.")
        self._querier.close_connection()

# --- Gemini Client ---
class GeminiClient:
    """Handles interactions with the Google Gemini API."""
    def __init__(self, config: RAGConfig):
        self.config = config
        genai.configure(api_key=config.gemini_api_key)
        self.base_generation_config = types.GenerationConfig(
            temperature=config.gemini_temperature,
            max_output_tokens=config.max_output_tokens
        )
        logging.info(f"Gemini client configured for model '{config.gemini_model_name}'.")

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> Optional[str]:
        """Generates content using the configured Gemini model."""
        try:
            model = genai.GenerativeModel(
                self.config.gemini_model_name,
                generation_config=self.base_generation_config,
                system_instruction=system_instruction
            )
            response = model.generate_content(prompt)
            # Check for prompt feedback even on success, might contain warnings
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                 logging.warning(f"Gemini Prompt Feedback: {response.prompt_feedback}")
            return response.text
        except types.generation_types.BlockedPromptException as bpe:
             logging.error(f"Prompt blocked by Gemini safety settings: {bpe}")
             print("The request was blocked by the content safety filter. Check logs.")
        except types.generation_types.StopCandidateException as sce:
             logging.error(f"Generation stopped unexpectedly by Gemini: {sce}")
             print(sce.partial_text if hasattr(sce, 'partial_text') else "Generation stopped unexpectedly.")
             # Return partial text if available
             return sce.partial_text if hasattr(sce, 'partial_text') else None
        except Exception as e:
            logging.error(f"Error generating content with Gemini: {e}")
            # Attempt to log prompt feedback if the response object exists
            # Note: 'response' might not be defined if the exception occurred early
            # This part might need refinement based on Gemini SDK behavior
            if 'response' in locals() and hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                logging.warning(f"Gemini Prompt Feedback: {response.prompt_feedback}")
            print("An error occurred during content generation. Check logs.")
        return None

# --- Context Formatter ---
class ContextFormatter:
    """Formats search results into a context string."""
    def __init__(self, config: RAGConfig):
        self.config = config

    def format(self, search_results: List[Dict[str, Any]]) -> str:
        context_str = ""
        current_word_count = 0
        added_doc_texts = set()

        # Sort results by confidence/similarity score if available
        sorted_results = sorted(search_results,
                               key=lambda x: x.get("score", 0),
                               reverse=True)

        for doc in sorted_results:
            doc_text = doc.get(self.config.doc_text_field, '')
            if not doc_text or doc_text in added_doc_texts:
                continue

            # Try to extract additional metadata if available
            metadata_lines = []
            for field in ["title", "author", "date", "topic", "category"]:
                if field in doc and doc[field]:
                    metadata_lines.append(f"{field.capitalize()}: {doc[field]}")

            metadata_str = ""
            if metadata_lines:
                metadata_str = "Metadata: " + "; ".join(metadata_lines) + "\n"

            content_line = f"Content: {doc_text}\n\n"
            text_to_add = metadata_str + content_line
            words_to_add = len(text_to_add.split())

            if current_word_count + words_to_add <= self.config.max_context_words:
                context_str += text_to_add
                current_word_count += words_to_add
                added_doc_texts.add(doc_text)
            else:
                if current_word_count < self.config.max_context_words: # Add truncation notice if space allows
                     truncation_notice = "Content: [Content truncated due to length limit]\n\n"
                     notice_words = len(truncation_notice.split())
                     if current_word_count + notice_words <= self.config.max_context_words:
                         context_str += truncation_notice
                         current_word_count += notice_words
                logging.warning(f"Context word limit ({self.config.max_context_words} words) reached. Some docs truncated.")
                break

        if not context_str:
            logging.warning("No valid content found in search results to format.")
        return context_str.strip()

# --- RAG Application Class ---
class RAGApplication:
    """Orchestrates the RAG process."""
    def __init__(self):
        logging.info("Initializing RAG Application...")
        self.config = RAGConfig()
        if not self.config.validate():
            raise ValueError("Configuration validation failed. Check logs.")

        self.prompt_loader = PromptLoader()
        self.system_prompt = self.prompt_loader.load(self.config.system_prompt_file)
        self.sub_query_prompt_template = self.prompt_loader.load(self.config.sub_query_prompt_file)

        if not self.system_prompt:
             logging.error(f"System prompt failed to load from {self.config.system_prompt_file}. Cannot proceed.")
             print(f"System prompt failed to load from {self.config.system_prompt_file}. Check logs.")
             raise ValueError("System prompt is essential and failed to load.")
        if not self.sub_query_prompt_template:
             logging.warning(f"Sub-query prompt template failed to load from {self.config.sub_query_prompt_file}. Sub-query generation will be skipped.")
             # Continue without sub-queries

        self.retriever = MongoRetriever(self.config)
        self.gemini_client = GeminiClient(self.config)
        self.formatter = ContextFormatter(self.config)
        logging.info("RAG Application initialized successfully.")

    def _generate_sub_queries(self, user_query: str) -> List[str]:
        """Generates sub-queries using the LLM."""
        if not self.sub_query_prompt_template:
            logging.warning("Skipping sub-query generation as the template was not loaded.")
            return []

        logging.info("Generating sub-queries...")
        prompt = self.sub_query_prompt_template.format(
            num_queries=self.config.num_sub_queries,
            user_query=user_query
        )
        response_text = self.gemini_client.generate(prompt)

        if not response_text:
            logging.warning("LLM did not return text for sub-queries.")
            return []

        # Basic parsing: assumes numbered list format
        raw_queries = response_text.strip().split('\n')
        sub_queries = [re.sub(r'^\d+\.\s*', '', q).strip() for q in raw_queries if q.strip()]

        if not sub_queries:
            logging.warning("LLM did not generate usable sub-queries. Using original query only.")
            return []

        logging.info("Generated sub-queries:")
        for i, sq in enumerate(sub_queries):
            logging.info(f"{i+1}. {sq}")
        return sub_queries

    def _retrieve_and_deduplicate_docs(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieves documents for multiple queries, aiming to add up to
        'num_retrieved_docs' unique documents for each query that were not
        retrieved by previous queries in the list.
        """
        all_unique_results = []
        processed_doc_ids = set() # Use doc ID for primary deduplication
        processed_doc_texts = set() # Fallback using text content
        target_docs_per_query = self.config.num_retrieved_docs
        # Fetch more candidates per query to increase chances of finding unique ones
        fetch_multiplier = 3 # Adjust as needed (higher means more initial results)
        fetch_k = target_docs_per_query * fetch_multiplier

        logging.info(f"Retrieving documents for {len(queries)} queries, aiming for up to {target_docs_per_query} unique new docs per query.")

        for i, query in enumerate(queries):
            logging.info(f"  Query {i+1}/{len(queries)}: '{query}' (Fetching top {fetch_k} candidates)")
            # Fetch potentially more documents than target_docs_per_query
            search_results = self.retriever.search(query, top_k=fetch_k)

            if not search_results:
                logging.info(f"    No documents found for this query.")
                continue

            unique_docs_added_for_this_query = 0
            docs_to_add_this_round = []

            # Iterate through the fetched results for the current query
            for doc in search_results:
                # Stop if we've already collected the target number of unique docs for this specific query
                if unique_docs_added_for_this_query >= target_docs_per_query:
                    break

                doc_id = doc.get(self.config.doc_id_field)
                doc_text = doc.get(self.config.doc_text_field)

                # Check if this document (by ID or text) has already been processed from a *previous* query
                is_processed = False
                if doc_id and doc_id in processed_doc_ids:
                    is_processed = True
                elif not doc_id and doc_text and doc_text in processed_doc_texts:
                    # Only consider text match if ID is missing
                    is_processed = True
                    logging.debug(f"Document missing ID matched existing text content: {doc_text[:50]}...") # More detailed log

                # If it's a new, unique document not seen before
                if not is_processed:
                    docs_to_add_this_round.append(doc)
                    unique_docs_added_for_this_query += 1

                    # Mark this document as processed for subsequent queries
                    if doc_id:
                        processed_doc_ids.add(doc_id)
                        if doc_text: # Keep text set consistent even if ID exists
                             processed_doc_texts.add(doc_text)
                    elif doc_text:
                        processed_doc_texts.add(doc_text)
                        # Log warning if ID was missing when adding based on text
                        logging.warning(f"Document missing '{self.config.doc_id_field}'. Added based on unique text content.")


            logging.info(f"    Fetched {len(search_results)} candidates, added {len(docs_to_add_this_round)} unique documents for this query.")
            all_unique_results.extend(docs_to_add_this_round)

        logging.info(f"Total unique documents collected across all queries: {len(all_unique_results)}")
        # Note: The final list `all_unique_results` contains documents unique across all queries.
        # The order reflects the order of queries and the relevance ranking *within* each query's fetch.
        # A final re-ranking of `all_unique_results` might be considered if needed.
        return all_unique_results

    def run(self):
        """Executes the main RAG workflow."""
        try:
            # 1. Get User Query
            user_query = input("Enter your question: ")
            if not user_query:
                logging.warning("No query entered. Exiting.")
                print("No query entered.")
                return

            # 2. Generate Sub-Queries
            sub_queries = self._generate_sub_queries(user_query)

            # 3. Retrieve Documents
            all_queries = [user_query] + sub_queries
            all_search_results = self._retrieve_and_deduplicate_docs(all_queries)

            # 4. Format Context
            if not all_search_results:
                logging.warning("Could not retrieve any relevant documents for any query.")
                context_str = "No relevant documents found."
                print("Could not retrieve any relevant documents to answer the question.")
                # Decide if you want to generate an answer without context
                # For now, we proceed but the LLM will know no docs were found.
            else:
                logging.info(f"Retrieved a total of {len(all_search_results)} unique documents.")
                logging.info("Formatting context...")
                context_str = self.formatter.format(all_search_results)
                if not context_str:
                     logging.warning("Failed to format context from retrieved documents.")
                     context_str = "Error formatting context." # Provide indication to LLM
                     print("An error occurred preparing context. The answer might be less accurate.")

            # 5. Prepare Final Prompt & Generate Response
            final_prompt = f"Question: {user_query}"
            system_instruction = self.system_prompt.format(context=context_str)

            logging.info("Generating final answer using Gemini...")
            final_answer = self.gemini_client.generate(
                prompt=final_prompt,
                system_instruction=system_instruction
            )

            # 6. Print LLM Response
            print("\n--- Answer ---")
            if final_answer is not None:
                print(final_answer)
            else:
                # Error messages during generation are printed by GeminiClient
                print("[No answer generated due to previous errors]")
            print("--------------")

        except Exception as e:
            logging.exception(f"An unexpected error occurred in RAGApplication.run: {e}")
            print("An unexpected error occurred. Check logs for details.")
        finally:
            # 7. Close Connections
            if hasattr(self, 'retriever') and self.retriever:
                self.retriever.close()
            logging.info("Script finished.")
            print(f"Log file saved to: {log_filename}")

# --- Main Execution ---
if __name__ == "__main__":
    try:
        app = RAGApplication()
        app.run()
    except ValueError as ve:
        # Handles initialization errors (e.g., missing config, essential prompts)
        logging.critical(f"Application initialization failed: {ve}")
        print(f"Application initialization failed: {ve}")
    except Exception as e:
        # Catch-all for unexpected errors during instantiation
        logging.critical(f"Failed to create RAGApplication: {e}", exc_info=True)
        print(f"An critical error occurred during setup: {e}")