import os
import re
import google.generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv
from mongo_querier import MongoQuerier  # Import directly
import logging
import datetime
from typing import List, Dict, Optional, Any
import json # Added for parsing sub-queries
import html 

# --- PDF Generation ---
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: reportlab library not found. PDF generation will be disabled.")
    print("Install it using: pip install reportlab")

# --- Basic Logging Setup ---
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
log_filename = os.path.join(log_directory, f"short_notes_gen_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_filename,
    filemode='w'
)

# --- Configuration Class ---
class ShortNotesConfig:
    """Loads and stores configuration parameters for short notes generation."""
    def __init__(self):
        load_dotenv()
        self.mongo_uri: Optional[str] = os.getenv("MONGO_URI")
        self.gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
        self.db_name: str = "upsc-database"  # Or your specific DB
        self.collection_name: str = "documents-embeddings" # Or your specific collection
        self.embedding_model: str = 'all-MiniLM-L6-v2' # Match model used for embedding
        self.gemini_model_name: str = "models/gemini-2.0-flash" # Or another suitable model
        self.num_sub_queries: int = 5 # Number of sub-queries to generate
        self.num_retrieved_docs_per_query: int = 15 # How many docs to fetch per query to increase unique doc pool
        self.max_context_words: int = 10000 # Limit context size for LLM
        self.max_output_tokens: int = 10000 # Max tokens for Gemini output
        self.gemini_temperature: float = 0.3 # Controls creativity vs factualness
        self.output_directory: str = "short_notes"
        self.doc_text_field: str = 'text' # Field containing document text in MongoDB
        self.doc_id_field: str = '_id' # Field containing the unique document ID in MongoDB

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
        if not REPORTLAB_AVAILABLE:
            logging.error("reportlab library is required for PDF generation but not found.")
            # Allow script to run but log error, PDF generation will fail later
            # return False # Uncomment this if PDF generation is strictly required
        return True

# --- Gemini Client (Adapted for Notes) ---
class GeminiClient:
    """Handles interactions with the Google Gemini API."""
    def __init__(self, config: ShortNotesConfig):
        self.config = config
        genai.configure(api_key=config.gemini_api_key)
        self.generation_config = types.GenerationConfig(
            temperature=config.gemini_temperature,
            max_output_tokens=config.max_output_tokens
        )
        self.sub_query_generation_config = types.GenerationConfig( # Separate config for sub-queries if needed
            temperature=0.5, # Slightly more creative for query generation
            max_output_tokens=500 # Lower token limit for sub-queries
        )
        logging.info(f"Gemini client configured for model '{config.gemini_model_name}'.")

    def generate_sub_queries(self, topic: str) -> Optional[List[str]]:
        """Generates sub-queries related to the main topic."""
        system_instruction = f"""You are an expert researcher. Given a main topic, break it down into {self.config.num_sub_queries} specific and diverse sub-topics or questions that would be useful for searching a database to gather comprehensive information.
Focus on different facets, key aspects, related concepts, or specific questions about the topic.
Return the sub-queries as a JSON list of strings. For example: ["sub-query 1", "sub-query 2", "sub-query 3", "sub-query 4", "sub-query 5"]

**Main Topic:** {topic}
---
Generate the JSON list of sub-queries now:"""

        prompt = f"Generate {self.config.num_sub_queries} sub-queries for the topic: {topic}" # Simple user prompt

        try:
            model = genai.GenerativeModel(
                self.config.gemini_model_name,
                generation_config=self.sub_query_generation_config, # Use specific config
                system_instruction=system_instruction
            )
            logging.info(f"Generating sub-queries for topic: {topic}")
            response = model.generate_content(prompt)

            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                 logging.warning(f"Gemini Sub-query Prompt Feedback: {response.prompt_feedback}")

            if not response.parts:
                 if response.candidates and response.candidates[0].finish_reason != 'STOP':
                     logging.error(f"Sub-query generation stopped due to reason: {response.candidates[0].finish_reason}")
                     print(f"Sub-query generation failed. Reason: {response.candidates[0].finish_reason}. Check logs.")
                     return None
                 else: # Likely blocked
                     logging.error("Sub-query generation blocked or failed.")
                     print("Sub-query generation failed or was blocked. Check logs.")
                     return None

            # Attempt to parse the response as JSON
            try:
                # Clean potential markdown code block fences
                cleaned_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
                sub_queries = json.loads(cleaned_text)
                if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
                    logging.info(f"Successfully generated {len(sub_queries)} sub-queries.")
                    return sub_queries[:self.config.num_sub_queries] # Ensure we don't exceed the requested number
                else:
                    logging.error(f"Gemini returned sub-queries in unexpected format: {sub_queries}")
                    print("Failed to parse sub-queries from Gemini response. Check logs.")
                    return None
            except json.JSONDecodeError as jde:
                logging.error(f"Failed to decode JSON response for sub-queries: {jde}. Response text: {response.text}")
                print("Failed to parse sub-queries from Gemini response (JSON decode error). Check logs.")
                return None
            except Exception as e:
                 logging.error(f"Error processing sub-query response: {e}. Response text: {response.text}", exc_info=True)
                 print("An error occurred while processing sub-queries. Check logs.")
                 return None

        except types.generation_types.BlockedPromptException as bpe:
             logging.error(f"Sub-query prompt blocked by Gemini safety settings: {bpe}")
             print("The sub-query request was blocked by the content safety filter. Check logs.")
        except types.generation_types.StopCandidateException as sce:
             logging.error(f"Sub-query generation stopped unexpectedly by Gemini: {sce}")
             print("Sub-query generation stopped unexpectedly.")
        except Exception as e:
            logging.error(f"Error generating sub-queries with Gemini: {e}", exc_info=True)
            print("An error occurred during sub-query generation. Check logs.")
        return None


    def generate_short_note(self, topic: str, context: str) -> Optional[str]:
        """Generates a short note using the configured Gemini model."""
        system_instruction = f"""You are an expert UPSC exam notes creator. Your task is to generate concise, comprehensive, and well-structured study notes on the given topic.

Use the provided context below, which contains relevant information retrieved from various sources based on the main topic and related sub-queries.

Follow these UPSC note-making principles:
1. Focus on core concepts, key facts, and important examples only
2. Use headings, subheadings, and bullet points for clear organization
3. Integrate static subject knowledge with relevant current affairs
4. Highlight keywords and important definitions
5. Include brief illustrative examples where helpful
6. Draw connections between related concepts
7. Create notes that serve both Prelims (factual recall) and Mains (analytical depth) exam needs

**Example Note Format:**
# MAIN TOPIC
## Key Concept 1
- Important point about this concept
- Another important point
- **Current Update:** Recent development related to this concept

## Key Concept 2
- Definition and significance
- Important facts and figures
- Relevant examples

**Main Topic:** {topic}

**Context:**
{context}
---
Generate the UPSC short note now:"""

        # Few-shot prompt to guide the model's output structure
        prompt = f"""Generate comprehensive UPSC study notes on: {topic}

Example of good UPSC notes structure:
1. Start with core definitions and background
2. Organize information under clear headings
3. Use bullet points for key facts
4. Highlight important terms in bold
5. Include current affairs developments 
6. Add a brief conclusion
"""

        try:
            model = genai.GenerativeModel(
                self.config.gemini_model_name,
                generation_config=self.generation_config,
                system_instruction=system_instruction # Use system instruction for detailed guidance
            )
            logging.info(f"Generating short note for topic: {topic}")
            response = model.generate_content(prompt) # Pass only the simple prompt here

            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                 logging.warning(f"Gemini Prompt Feedback: {response.prompt_feedback}")

            # Check if the response was blocked or stopped early
            if not response.parts:
                 if response.candidates and response.candidates[0].finish_reason != 'STOP':
                     logging.error(f"Generation stopped due to reason: {response.candidates[0].finish_reason}")
                     print(f"Note generation failed or was stopped. Reason: {response.candidates[0].finish_reason}. Check logs.")
                     return None
                 else: # Likely blocked
                     logging.error("Generation blocked or failed with no specific reason provided in response.")
                     print("Note generation failed or was blocked. Check logs.")
                     return None

            return response.text

        except types.generation_types.BlockedPromptException as bpe:
             logging.error(f"Prompt blocked by Gemini safety settings: {bpe}")
             print("The request was blocked by the content safety filter. Check logs.")
        except types.generation_types.StopCandidateException as sce:
             logging.error(f"Generation stopped unexpectedly by Gemini: {sce}")
             print(sce.partial_text if hasattr(sce, 'partial_text') else "Generation stopped unexpectedly.")
             return sce.partial_text if hasattr(sce, 'partial_text') else None # Return partial if available
        except Exception as e:
            logging.error(f"Error generating content with Gemini: {e}", exc_info=True)
            if 'response' in locals() and hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                logging.warning(f"Gemini Prompt Feedback: {response.prompt_feedback}")
            print("An error occurred during note generation. Check logs.")
        return None

# --- Context Formatter (Simplified) ---
class ContextFormatter:
    """Formats search results into a context string."""
    def __init__(self, config: ShortNotesConfig):
        self.config = config

    def format(self, search_results: List[Dict[str, Any]]) -> str:
        """Formats a list of unique documents into a context string."""
        context_str = ""
        current_word_count = 0
        # No need for text deduplication here if uniqueness is handled by ID upstream
        # added_doc_texts = set()

        # Sort by score if available, otherwise use retrieval order
        # Note: Scores might be less meaningful when combined from different queries
        # Consider sorting later or just using retrieval order. Keeping sort for now.
        sorted_results = sorted(search_results, key=lambda x: x.get("score", 0), reverse=True)

        logging.info(f"Formatting context from {len(sorted_results)} unique retrieved documents.")
        for i, doc in enumerate(sorted_results):
            doc_text = doc.get(self.config.doc_text_field, '')
            if not doc_text: # Skip docs with empty text content
                logging.debug(f"Skipping document with empty text (Doc Index {i}). ID: {doc.get(self.config.doc_id_field, 'N/A')}")
                continue

            # Simple formatting: Add separator between docs
            # Include score and potentially source if available and desired
            score = doc.get("score", "N/A")
            source_info = doc.get("metadata", {}).get("source", "Unknown Source") # Example if metadata exists
            text_to_add = f"--- Document {i+1} (Score: {score:.4f}, Source: {source_info}) ---\n{doc_text}\n\n"
            words_to_add = len(text_to_add.split())

            if current_word_count + words_to_add <= self.config.max_context_words:
                context_str += text_to_add
                current_word_count += words_to_add
                # added_doc_texts.add(doc_text) # Not needed if unique by ID
            else:
                logging.warning(f"Context word limit ({self.config.max_context_words} words) reached. Stopping context assembly. Included {i} documents.")
                break # Stop adding documents once limit is hit

        if not context_str:
            logging.warning("No valid content found in unique search results to format context.")
            return "No relevant information found in the database." # Inform LLM

        logging.info(f"Formatted context generated with approx. {current_word_count} words from {i+1 if current_word_count > 0 else 0} documents.")
        return context_str.strip()

# --- PDF Generator ---
class PDFGenerator:
    """Generates a PDF document from text content."""
    def __init__(self, config: ShortNotesConfig):
        # ... (init remains the same) ...
        self.config = config
        os.makedirs(self.config.output_directory, exist_ok=True)
    def _preprocess_text(self, text: str) -> str:
        """Applies basic XML markup for ReportLab (e.g., bold)."""
        # Escape HTML special characters first to avoid conflicts with our markup
        text = html.escape(text)
        # Handle bold (**text**) -> <b>text</b> (Now applied *after* escaping)
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        # Handle italics (*text* or _text_) -> <i>text</i>
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        text = re.sub(r'_(.*?)_', r'<i>\1</i>', text)
        # Replace escaped entities for bold/italic tags back to symbols
        text = text.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
        text = text.replace('&lt;i&gt;', '<i>').replace('&lt;/i&gt;', '</i>')
        return text
    def generate(self, topic: str, content: str) -> Optional[str]:
        """Creates a PDF file with the given topic and content, interpreting basic markdown."""
        if not REPORTLAB_AVAILABLE:
            logging.error("Cannot generate PDF: reportlab library is not installed.")
            return None

        safe_topic = re.sub(r'[^\w\-_\. ]', '_', topic)
        filename = os.path.join(self.config.output_directory, f"{safe_topic}.pdf")

        try:
            doc = SimpleDocTemplate(filename, pagesize=letter,
                                    leftMargin=inch, rightMargin=inch,
                                    topMargin=inch, bottomMargin=inch)
            styles = getSampleStyleSheet()
            styles['BodyText'].leading = 14
            # Add custom heading styles if needed (optional, but good practice)
            styles.add(ParagraphStyle(name='H1', parent=styles['h1'], spaceAfter=0.12*inch))
            styles.add(ParagraphStyle(name='H2', parent=styles['h2'], spaceAfter=0.08*inch))
            styles.add(ParagraphStyle(name='H3', parent=styles['h3'], spaceAfter=0.06*inch))


            story = []
            current_style = styles['BodyText'] # Keep track of the current style

            # Main Title
            try:
                story.append(Paragraph(f"Short Note: {html.escape(topic)}", styles['H1'])) # Escape topic too
                story.append(Spacer(1, 0.15*inch))
            except Exception as pe:
                logging.error(f"PDF Error: Failed to create main title paragraph for topic '{topic}': {pe}", exc_info=True)
                # Decide whether to continue or fail here
                # return None # Option: Fail PDF generation entirely

            # Content Parsing
            lines = content.strip().split('\n')
            for i, line_content in enumerate(lines):
                original_line = line_content # Keep original for indentation check
                line = line_content.strip()

                if not line:
                    # Add a small spacer for empty lines to represent paragraph breaks visually
                    # story.append(Spacer(1, 0.1*inch)) # Optional: uncomment to add space for blank lines
                    continue

                try:
                    # Preprocess *after* structure detection for headings, but *before* for others
                    if line.startswith('# '):
                        text = self._preprocess_text(line[2:].strip())
                        para = Paragraph(text, styles['H1'])
                        story.append(para)
                        # Spacer is handled by style's spaceAfter now
                    elif line.startswith('## '):
                        text = self._preprocess_text(line[3:].strip())
                        para = Paragraph(text, styles['H2'])
                        story.append(para)
                    elif line.startswith('### '):
                        text = self._preprocess_text(line[4:].strip())
                        para = Paragraph(text, styles['H3'])
                        story.append(para)
                    elif line.startswith('* ') or line.startswith('- '):
                        text_content = self._preprocess_text(line[2:].strip())
                        leading_spaces = len(original_line) - len(original_line.lstrip(' '))
                        indent_level = leading_spaces // 2
                        indent_amount = 0.25 * indent_level * inch

                        bullet_style_name = f'bullet_level_{indent_level}'
                        if bullet_style_name not in styles:
                            # Create style if it doesn't exist
                            styles.add(ParagraphStyle(
                                name=bullet_style_name,
                                parent=styles['BodyText'],
                                leftIndent=indent_amount,
                                firstLineIndent=-0.1 * inch, # Hanging indent
                                leading=12,
                                spaceAfter=0.03*inch # Space after list item
                            ))
                        current_style = styles[bullet_style_name]

                        bullet_char = "•"
                        if indent_level == 1: bullet_char = "◦"
                        elif indent_level >= 2: bullet_char = "▪"

                        # Use &bull; entity for standard bullet, others as is
                        bullet_text = f"{bullet_char}&nbsp;&nbsp;{text_content}"
                        para = Paragraph(bullet_text, current_style)
                        story.append(para)
                        # Spacer is handled by style's spaceAfter now
                    else:
                        # Regular paragraph
                        text = self._preprocess_text(line)
                        para = Paragraph(text, styles['BodyText'])
                        story.append(para)
                        story.append(Spacer(1, 0.05*inch)) # Keep spacer for paragraph separation

                except Exception as pe:
                    logging.error(f"PDF Error: Failed processing line {i+1}. Content: '{line_content}'. Error: {pe}", exc_info=True)
                    # Option 1: Skip the problematic line
                    story.append(Paragraph(f"<i>[Error processing line {i+1}]</i>", styles['BodyText']))
                    # Option 2: Fail PDF generation
                    # print(f"Error generating PDF content on line {i+1}. Check logs.")
                    # return None

            logging.info(f"Building PDF document: {filename}")
            doc.build(story)
            logging.info(f"Successfully generated PDF: {filename}")
            return filename

        except Exception as e:
            # Catch errors during doc.build() or other setup issues
            logging.error(f"Error generating PDF '{filename}': {e}", exc_info=True)
            print(f"Error generating PDF for topic '{topic}'. Check logs.")
            # Attempt to clean up potentially partially created file
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                    logging.info(f"Removed partially created PDF: {filename}")
                except OSError as oe:
                    logging.error(f"Failed to remove partially created PDF '{filename}': {oe}")
            return None

# --- Short Notes Generator Application ---
class ShortNotesGeneratorApp:
    """Orchestrates the short notes generation process."""
    def __init__(self):
        logging.info("Initializing Short Notes Generator Application...")
        self.config = ShortNotesConfig()
        if not self.config.validate():
            raise ValueError("Configuration validation failed. Check logs.")

        # Initialize components
        self.mongo_querier = MongoQuerier(
            mongo_uri=self.config.mongo_uri,
            db_name=self.config.db_name,
            collection_name=self.config.collection_name,
            model_name=self.config.embedding_model
        )
        self.gemini_client = GeminiClient(self.config)
        self.formatter = ContextFormatter(self.config)
        self.pdf_generator = PDFGenerator(self.config)
        logging.info("Short Notes Generator Application initialized successfully.")

    def run(self):
        """Executes the main short notes generation workflow."""
        try:
            # 1. Get Topic from User
            topic = input("Enter the topic for the short note: ")
            if not topic:
                logging.warning("No topic entered. Exiting.")
                print("No topic entered.")
                return

            # 2. Generate Sub-queries using Gemini
            print("Generating relevant sub-queries...")
            sub_queries = self.gemini_client.generate_sub_queries(topic)
            if not sub_queries:
                logging.warning(f"Could not generate sub-queries for topic: '{topic}'. Proceeding with main topic only.")
                print("Warning: Could not generate sub-queries. Proceeding with the main topic only.")
                queries_to_search = [topic]
            else:
                logging.info(f"Generated sub-queries: {sub_queries}")
                print(f"Generated {len(sub_queries)} sub-queries.")
                # Combine main topic with sub-queries for searching
                queries_to_search = [topic] + sub_queries

            # 3. Retrieve Documents from MongoDB for all queries
            print(f"Retrieving documents for '{topic}' and related sub-queries...")
            all_retrieved_docs = [] # Renamed from all_unique_docs
            # seen_doc_ids = set() # Removed uniqueness check
            total_retrieved_count = 0

            for i, query in enumerate(queries_to_search):
                logging.info(f"Searching documents for query {i+1}/{len(queries_to_search)}: '{query}'")
                try:
                    search_results = self.mongo_querier.search_similar_documents(
                        query,
                        top_k=self.config.num_retrieved_docs_per_query
                    )

                    if search_results:
                        docs_added_for_query = len(search_results)
                        total_retrieved_count += docs_added_for_query
                        all_retrieved_docs.extend(search_results) # Add all results directly
                        logging.info(f"Added {docs_added_for_query} documents from query '{query}'. Total documents collected: {len(all_retrieved_docs)}")
                        # Removed uniqueness check logic
                        # new_docs_for_query = 0
                        # for doc in search_results:
                        #     doc_id = doc.get(self.config.doc_id_field)
                        #     if doc_id is None:
                        #         logging.warning(f"Document missing ID field ('{self.config.doc_id_field}'). Skipping.")
                        #         continue
                        #     # Convert ObjectId to str if necessary for the set
                        #     doc_id_str = str(doc_id)
                        #     # if doc_id_str not in seen_doc_ids: # Removed uniqueness check
                        #     #     seen_doc_ids.add(doc_id_str)
                        #     all_retrieved_docs.append(doc)
                        #     new_docs_for_query += 1
                        # logging.info(f"Added {new_docs_for_query} new unique documents from query '{query}'. Total unique docs: {len(all_retrieved_docs)}")
                    elif search_results is None: # Indicates an error during search
                         logging.error(f"Error retrieving documents for query: '{query}'")
                         print(f"Warning: Failed to retrieve documents for query '{query}'. Check logs.")
                         # Continue with other queries
                    else: # Empty list returned
                        logging.info(f"No documents found for query: '{query}'")

                except Exception as e:
                    logging.exception(f"Error during document search for query '{query}': {e}")
                    print(f"An error occurred while searching for query '{query}'. Check logs.")
                    # Continue with other queries

            logging.info(f"Total documents retrieved across all queries: {total_retrieved_count}") # Updated log message
            print(f"Found {len(all_retrieved_docs)} relevant documents (including potential duplicates).") # Updated print message

            if not all_retrieved_docs:
                logging.warning(f"No documents found in MongoDB for topic '{topic}' or its sub-queries.")
                print("No relevant documents found in the database for this topic or related sub-queries.")
                # Let context formatter handle empty results, LLM will be informed.

            # Deduplicate documents based on document ID
            seen_doc_ids = set()
            unique_docs = []
            for doc in all_retrieved_docs:
                doc_id = doc.get(self.config.doc_id_field)
                if doc_id is None:
                    logging.warning(f"Document missing ID field ('{self.config.doc_id_field}'). Skipping.")
                    continue
                # Convert ObjectId to str if necessary
                doc_id_str = str(doc_id)
                if doc_id_str not in seen_doc_ids:
                    seen_doc_ids.add(doc_id_str)
                    unique_docs.append(doc)
            
            logging.info(f"Deduplicated {len(all_retrieved_docs)} documents to {len(unique_docs)} unique documents.")
            print(f"Deduplicating documents: {len(unique_docs)} unique documents out of {len(all_retrieved_docs)} total.")

            # 4. Format Context from retrieved documents
            context_str = self.formatter.format(unique_docs)  # Pass the deduplicated list

            # 5. Generate Short Note using Gemini
            print("Generating short note using Gemini...")
            logging.info("Generating short note using Gemini...")
            note_content = self.gemini_client.generate_short_note(topic, context_str)

            # 6. Save Note as PDF
            if note_content:
                print("\n--- Generated Short Note ---")
                print(note_content)
                print("--------------------------")
                logging.info("Saving generated note to PDF...")
                pdf_filepath = self.pdf_generator.generate(topic, note_content)
                if pdf_filepath:
                    print(f"Short note saved successfully to: {pdf_filepath}")
                else:
                    print("Failed to save the note as a PDF file. Check logs.")
            else:
                # Specific message if context was empty vs. LLM failure
                if not context_str or context_str == "No relevant information found in the database.":
                     print("\n[Could not generate a short note because no relevant information was found in the database.]")
                     logging.warning(f"Note generation skipped for topic '{topic}' due to lack of context.")
                else:
                    print("\n[Could not generate a short note for this topic. An error occurred during generation or the request was blocked.]")
                    logging.warning(f"Failed to generate note content for topic: {topic}")

        except Exception as e:
            logging.exception(f"An unexpected error occurred in ShortNotesGeneratorApp.run: {e}")
            print("An unexpected error occurred. Check logs for details.")
        finally:
            # 7. Close Connections
            if hasattr(self, 'mongo_querier') and self.mongo_querier:
                self.mongo_querier.close_connection()
            logging.info("Script finished.")
            print(f"\nLog file saved to: {log_filename}")

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure reportlab is checked early if needed
    if not REPORTLAB_AVAILABLE:
         print("Warning: reportlab library not found. PDF generation will be disabled.")
         # print("Exiting: reportlab library is required for PDF generation but is not installed.")
         # exit() # Uncomment to force exit if PDF is mandatory

    try:
        app = ShortNotesGeneratorApp()
        app.run()
    except ValueError as ve:
        # Handles initialization errors (e.g., missing config)
        logging.critical(f"Application initialization failed: {ve}")
        print(f"Application initialization failed: {ve}")
    except Exception as e:
        # Catch-all for unexpected errors during instantiation
        logging.critical(f"Failed to create ShortNotesGeneratorApp: {e}", exc_info=True)
        print(f"An critical error occurred during setup: {e}")
