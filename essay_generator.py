import os
import re
import google.generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv
from mongo_querier import MongoQuerier
import logging
import datetime
from typing import List, Dict, Optional, Any
import json
import html

# --- PDF Generation ---
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: reportlab library not found. PDF generation will be disabled.")
    print("Install it using: pip install reportlab")

# --- Basic Logging Setup ---
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
log_filename = os.path.join(log_directory, f"upsc_essay_gen_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_filename,
    filemode='w'
)

# --- UPSC Essay Config ---
class UPSCEssayConfig:
    """Loads and stores configuration parameters for the UPSC essay generator."""
    def __init__(self):
        load_dotenv()
        self.mongo_uri: Optional[str] = os.getenv("MONGO_URI")
        self.gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
        self.db_name: str = "upsc-database"
        self.collection_name: str = "documents-embeddings"
        self.embedding_model: str = 'all-MiniLM-L6-v2'
        self.gemini_model_name: str = "models/gemini-2.0-flash"
        self.num_sub_queries: int = 5
        self.num_retrieved_docs_per_query: int = 15
        self.max_context_words: int = 12000
        self.max_output_tokens: int = 50000
        self.gemini_temperature: float = 0.3
        self.output_directory: str = "upsc_essays"
        self.doc_id_field: str = '_id'
        self.doc_text_field: str = 'text'
        self.doc_filepath_field: str = 'file_path'
        
        # Define file paths for prompts
        self.prompts_dir: str = "adhikari-ai/prompts"
        os.makedirs(self.prompts_dir, exist_ok=True)
        
        # Create prompt files if they don't exist
        self.essay_prompt_file = os.path.join(self.prompts_dir, "upsc_essay_prompt.txt")
        self.sub_query_prompt_file = os.path.join(self.prompts_dir, "upsc_essay_sub_query_prompt.txt")
        
        self._create_default_prompts()
        
    def _create_default_prompts(self):
        """Creates default prompt files if they don't exist."""
        # Essay generation prompt
        if not os.path.exists(self.essay_prompt_file):
            essay_prompt = """You are an expert UPSC essay writer. Your task is to write a comprehensive, well-structured essay on the provided topic using the context provided. Follow these UPSC essay guidelines:

1. Structure: Create a clear introduction, well-developed body paragraphs, and a strong conclusion.
2. Introduction: Start with an engaging opening (quote, statistics, anecdote), provide context, and establish your main argument.
3. Body Paragraphs: Each paragraph should focus on one main idea with supporting evidence from the context.
4. Include diverse perspectives: Cover political, economic, social, technological, legal, environmental, and ethical aspects where relevant.
5. Evidence: Use concrete examples, data, and case studies from the provided context.
6. Counter-arguments: Acknowledge opposing viewpoints and address them respectfully.
7. Conclusion: Summarize main points, reiterate your thesis, and end with a thoughtful closing statement.
8. Language: Use clear, precise language. Avoid jargon or overly complex sentences.
9. **Word count: Strictly adhere to a word count between 2000 and 3000 words.**.

The context below contains information relevant to the topic: {topic}

Use this information to develop your essay, but don't simply copy it. Synthesize, integrate, and **expand significantly** upon the information in a coherent way to meet the required length.

Context:
{context}"""
            with open(self.essay_prompt_file, 'w') as f:
                f.write(essay_prompt)
            logging.info(f"Created default essay prompt file at {self.essay_prompt_file}")
        
        # Sub-query generation prompt
        if not os.path.exists(self.sub_query_prompt_file):
            sub_query_prompt = """Your task is to generate {num_queries} distinct sub-queries that will help gather comprehensive information for writing a UPSC essay on the topic: "{topic}".

For UPSC essays, we need to explore multiple dimensions of the topic, including:
1. Historical background and evolution
2. Current relevance and issues
3. Various stakeholders' perspectives
4. Economic implications
5. Social and cultural aspects
6. Political dimensions
7. Ethical considerations
8. International context and comparisons
9. Future trends and challenges
10. Potential solutions and recommendations

Generate {num_queries} search queries that would help retrieve relevant information covering different aspects of the topic. Make the queries specific enough to retrieve focused information but broad enough to capture diverse perspectives.

Format your response as a JSON array of strings, each representing one search query."""
            with open(self.sub_query_prompt_file, 'w') as f:
                f.write(sub_query_prompt)
            logging.info(f"Created default sub-query prompt file at {self.sub_query_prompt_file}")

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
        if not os.path.exists(self.essay_prompt_file):
            logging.error(f"Essay prompt file not found: {self.essay_prompt_file}")
            return False
        if not os.path.exists(self.sub_query_prompt_file):
            logging.error(f"Sub-query prompt file not found: {self.sub_query_prompt_file}")
            return False
        # Create output directory if it doesn't exist
        os.makedirs(self.output_directory, exist_ok=True)
        return True

# --- Gemini Client ---
class GeminiClient:
    """Handles interactions with the Google Gemini API for essay generation."""
    def __init__(self, config: UPSCEssayConfig):
        self.config = config
        genai.configure(api_key=config.gemini_api_key)
        self.generation_config = types.GenerationConfig(
            temperature=config.gemini_temperature,
            max_output_tokens=config.max_output_tokens
        )
        self.sub_query_generation_config = types.GenerationConfig(
            temperature=0.5,  # Slightly more creative for sub-queries
            max_output_tokens=500  # Lower token limit for sub-queries
        )
        # Load prompt templates
        try:
            with open(config.essay_prompt_file, 'r') as f:
                self.essay_prompt_template = f.read()
            with open(config.sub_query_prompt_file, 'r') as f:
                self.sub_query_prompt_template = f.read()
            logging.info("Successfully loaded prompt templates.")
        except Exception as e:
            logging.error(f"Error loading prompt files: {e}", exc_info=True)
            raise

    def generate_sub_queries(self, topic: str) -> Optional[List[str]]:
        """Generates sub-queries related to the main essay topic."""
        try:
            system_instruction = self.sub_query_prompt_template.format(
                num_queries=self.config.num_sub_queries,
                topic=topic
            )
        except KeyError as e:
            logging.error(f"Missing placeholder in sub_query prompt: {e}")
            print(f"Error: Placeholder {e} missing in sub_query prompt. Check the file.")
            return None

        prompt = f"Generate {self.config.num_sub_queries} sub-queries for the UPSC essay topic: {topic}"

        try:
            model = genai.GenerativeModel(
                self.config.gemini_model_name,
                generation_config=self.sub_query_generation_config,
                system_instruction=system_instruction
            )
            logging.info(f"Generating sub-queries for topic: {topic}")
            response = model.generate_content(prompt)

            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                logging.warning(f"Gemini Sub-query Prompt Feedback: {response.prompt_feedback}")

            if not response.parts:
                logging.error("Sub-query generation failed or was blocked.")
                print("Sub-query generation failed or was blocked. Check logs.")
                return None

            # Try to parse the response as JSON
            try:
                # Clean potential markdown code block fences
                cleaned_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
                sub_queries = json.loads(cleaned_text)
                if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
                    logging.info(f"Successfully generated {len(sub_queries)} sub-queries.")
                    return sub_queries[:self.config.num_sub_queries]
                else:
                    logging.error(f"Gemini returned sub-queries in unexpected format: {sub_queries}")
                    print("Failed to parse sub-queries from Gemini response. Check logs.")
                    return None
            except json.JSONDecodeError as jde:
                # Fallback: Try to extract queries using regex if JSON parsing fails
                try:
                    # Look for numbered list or quotes
                    queries = re.findall(r'\d+\.\s*"([^"]+)"', response.text)
                    if not queries:
                        queries = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|\Z)', response.text)
                    if queries:
                        logging.info(f"Parsed {len(queries)} sub-queries using regex fallback.")
                        return [q.strip() for q in queries[:self.config.num_sub_queries]]
                    else:
                        logging.error(f"Failed to extract sub-queries using fallback method")
                        return None
                except Exception as regex_e:
                    logging.error(f"Failed to parse sub-queries from text: {regex_e}")
                    return None
        except Exception as e:
            logging.error(f"Error generating sub-queries: {e}", exc_info=True)
            print("An error occurred during sub-query generation. Check logs.")
        return None

    def generate_essay(self, topic: str, context: str) -> Optional[str]:
        """Generates a UPSC essay using the configured Gemini model."""
        try:
            system_instruction = self.essay_prompt_template.format(
                topic=topic,
                context=context
            )
        except KeyError as e:
            logging.error(f"Missing placeholder in essay prompt: {e}")
            print(f"Error: Placeholder {e} missing in essay prompt. Check the file.")
            return None

        prompt = f"Write a comprehensive UPSC essay on the topic: {topic}"

        try:
            model = genai.GenerativeModel(
                self.config.gemini_model_name,
                generation_config=self.generation_config,
                system_instruction=system_instruction
            )
            logging.info(f"Generating essay for topic: {topic}")
            response = model.generate_content(prompt)

            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                logging.warning(f"Gemini Prompt Feedback: {response.prompt_feedback}")

            if not response.parts:
                logging.error("Essay generation failed or was blocked.")
                print("Essay generation failed or was blocked. Check logs.")
                return None

            essay_content = response.text
            
            # Check approximate word count
            word_count = len(essay_content.split())
            logging.info(f"Generated essay with approximately {word_count} words.")
            
            if word_count < 800:
                logging.warning(f"Essay is shorter than expected (only {word_count} words).")
                print(f"Warning: Generated essay is shorter than expected ({word_count} words). UPSC essays should be 1000-1200 words.")
            
            return essay_content

        except types.generation_types.BlockedPromptException as bpe:
            logging.error(f"Essay prompt blocked by Gemini safety settings: {bpe}")
            print("The essay request was blocked by the content safety filter. Check logs.")
        except types.generation_types.StopCandidateException as sce:
            logging.error(f"Essay generation stopped unexpectedly by Gemini: {sce}")
            print("Essay generation stopped unexpectedly.")
            return sce.partial_text if hasattr(sce, 'partial_text') else None
        except Exception as e:
            logging.error(f"Error generating essay with Gemini: {e}", exc_info=True)
            print("An error occurred during essay generation. Check logs.")
        return None

# --- Context Formatter ---
class ContextFormatter:
    """Formats search results into a context string for essay generation."""
    def __init__(self, config: UPSCEssayConfig):
        self.config = config

    def format(self, search_results: List[Dict[str, Any]]) -> str:
        """Formats a list of unique documents into a context string."""
        context_str = ""
        current_word_count = 0
        
        # Sort results by score if available
        sorted_results = sorted(search_results, key=lambda x: x.get("score", 0), reverse=True)

        logging.info(f"Formatting context from {len(sorted_results)} unique retrieved documents.")
        for i, doc in enumerate(sorted_results):
            doc_text = doc.get(self.config.doc_text_field, '')
            if not doc_text:
                continue

            # Format the document with metadata if available
            source_info = doc.get("file_path", "Unknown Source")
            text_to_add = f"--- Document {i+1} ---\nSource: {source_info}\n{doc_text}\n\n"
            words_to_add = len(text_to_add.split())

            if current_word_count + words_to_add <= self.config.max_context_words:
                context_str += text_to_add
                current_word_count += words_to_add
            else:
                logging.warning(f"Context word limit ({self.config.max_context_words} words) reached after {i} documents.")
                break

        if not context_str:
            logging.warning("No valid content found in search results to format context.")
            return "No relevant information found in the database."

        logging.info(f"Formatted context with approximately {current_word_count} words.")
        return context_str.strip()

# --- PDF Generator ---
class PDFGenerator:
    """Generates a PDF document from the essay content."""
    def __init__(self, config: UPSCEssayConfig):
        self.config = config
        os.makedirs(self.config.output_directory, exist_ok=True)

    def _preprocess_text(self, text: str) -> str:
        """Preprocesses text for PDF generation, handling formatting."""
        # Escape HTML special characters first
        text = html.escape(text)
        # Handle bold (**text**)
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        # Handle italics (*text* or _text_)
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        text = re.sub(r'_(.*?)_', r'<i>\1</i>', text)
        # Replace escaped entities for bold/italic tags back to symbols
        text = text.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
        text = text.replace('&lt;i&gt;', '<i>').replace('&lt;/i&gt;', '</i>')
        return text

    def generate(self, topic: str, content: str) -> Optional[str]:
        """Creates a PDF file with the UPSC essay."""
        if not REPORTLAB_AVAILABLE:
            logging.error("Cannot generate PDF: reportlab library is not installed.")
            return None

        # Create a safe filename from the topic
        safe_topic = re.sub(r'[^\w\-_\. ]', '_', topic)
        filename = os.path.join(self.config.output_directory, f"UPSC_Essay_{safe_topic}.pdf")

        try:
            doc = SimpleDocTemplate(filename, pagesize=letter,
                                    leftMargin=inch, rightMargin=inch,
                                    topMargin=inch, bottomMargin=inch)
            
            # Create custom styles for UPSC essay format
            styles = getSampleStyleSheet()
            
            # Title style
            title_style = ParagraphStyle(
                name='EssayTitle',
                parent=styles['Heading1'],
                fontSize=14,
                alignment=1,  # Center alignment
                spaceAfter=0.3*inch
            )
            
            # Body paragraph style
            body_style = ParagraphStyle(
                name='EssayBody',
                parent=styles['Normal'],
                fontSize=12,
                leading=16,  # Line spacing
                alignment=TA_JUSTIFY,  # Justified text
                firstLineIndent=0.25*inch,  # First line indent
                spaceAfter=0.1*inch  # Space after paragraph
            )
            
            # Heading styles
            heading1_style = ParagraphStyle(
                name='EssayH1',
                parent=styles['Heading2'],
                fontSize=13,
                spaceAfter=0.1*inch,
                spaceBefore=0.2*inch
            )
            
            heading2_style = ParagraphStyle(
                name='EssayH2',
                parent=styles['Heading3'],
                fontSize=12,
                spaceAfter=0.08*inch,
                spaceBefore=0.15*inch
            )

            # Start building the document
            story = []
            
            # Add title
            story.append(Paragraph(f"<b>Essay Topic: {html.escape(topic)}</b>", title_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Parse and format the content
            lines = content.strip().split('\n')
            current_style = body_style
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    if line.startswith('# ') or re.match(r'^[A-Z\s]+:$', line):
                        # Main heading
                        text = self._preprocess_text(line.lstrip('# '))
                        para = Paragraph(text, heading1_style)
                        story.append(Spacer(1, 0.1*inch))
                        story.append(para)
                    elif line.startswith('## ') or re.match(r'^[A-Z][a-z\s]+:$', line):
                        # Subheading
                        text = self._preprocess_text(line.lstrip('## '))
                        para = Paragraph(text, heading2_style)
                        story.append(para)
                    else:
                        # Regular paragraph
                        text = self._preprocess_text(line)
                        para = Paragraph(text, body_style)
                        story.append(para)
                except Exception as pe:
                    logging.error(f"PDF Error processing line: '{line}'. Error: {pe}", exc_info=True)
                    # Try a simpler version if processing fails
                    try:
                        story.append(Paragraph(html.escape(line), styles['Normal']))
                    except:
                        logging.error(f"Failed even with simple paragraph processing.")
            
            # Add word count at the end
            word_count = len(content.split())
            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph(f"<i>Word Count: Approximately {word_count} words</i>", 
                                  styles['Normal']))
            
            # Build the PDF
            logging.info(f"Building PDF document: {filename}")
            doc.build(story)
            logging.info(f"Successfully generated PDF: {filename}")
            return filename
            
        except Exception as e:
            logging.error(f"Error generating PDF '{filename}': {e}", exc_info=True)
            print(f"Error generating PDF for topic '{topic}'. Check logs.")
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                except:
                    pass
            return None

# --- UPSC Essay Generator Application ---
class UPSCEssayGenerator:
    """Main application for generating UPSC essays."""
    def __init__(self):
        logging.info("Initializing UPSC Essay Generator Application...")
        self.config = UPSCEssayConfig()
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
        logging.info("UPSC Essay Generator Application initialized successfully.")

    def retrieve_and_deduplicate_docs(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Retrieves documents for multiple queries and removes duplicates."""
        all_retrieved_docs = []
        seen_doc_ids = set()

        for i, query in enumerate(queries):
            logging.info(f"Searching documents for query {i+1}/{len(queries)}: '{query}'")
            try:
                search_results = self.mongo_querier.search_similar_documents(
                    query,
                    top_k=self.config.num_retrieved_docs_per_query
                )

                if search_results:
                    initial_seen_count = len(seen_doc_ids) # Track count before adding from this query
                    for doc in search_results:
                        # Use the configured doc_id_field
                        doc_id = doc.get(self.config.doc_id_field)
                        if doc_id is None:
                            # Add a warning if the ID field is missing
                            logging.warning(f"Document found without ID field '{self.config.doc_id_field}'. Skipping. Doc keys: {list(doc.keys())}")
                            continue

                        # Convert ObjectId to str if necessary for the set
                        doc_id_str = str(doc_id)
                        if doc_id_str not in seen_doc_ids:
                            seen_doc_ids.add(doc_id_str)
                            all_retrieved_docs.append(doc) # Add the document if its ID is new

                    # Correctly calculate and log the number of *new* unique documents added
                    new_docs_added = len(seen_doc_ids) - initial_seen_count
                    logging.info(f"Added {new_docs_added} new unique documents from query '{query}'. Total unique docs now: {len(all_retrieved_docs)}")

            except Exception as e:
                logging.exception(f"Error during document search for query '{query}': {e}")
                print(f"An error occurred while searching for query '{query}'. Check logs.")

        logging.info(f"Total unique documents retrieved: {len(all_retrieved_docs)}")
        return all_retrieved_docs

    def run(self):
        """Executes the main UPSC essay generation workflow."""
        try:
            print("====== UPSC ESSAY GENERATOR ======")
            print("This tool will help you generate a well-structured UPSC essay on your chosen topic.")
            
            # 1. Get Topic from User
            topic = input("Enter the UPSC essay topic: ")
            if not topic:
                logging.warning("No topic entered. Exiting.")
                print("No topic entered.")
                return
            
            # 2. Generate Sub-queries using Gemini
            print("\nGenerating relevant sub-queries to explore different aspects of the topic...")
            sub_queries = self.gemini_client.generate_sub_queries(topic)
            if not sub_queries:
                logging.warning(f"Could not generate sub-queries for topic: '{topic}'. Proceeding with main topic only.")
                print("Warning: Could not generate sub-queries. Proceeding with the main topic only.")
                queries_to_search = [topic]
            else:
                print("\nGenerated sub-queries to explore different aspects of your topic:")
                for i, query in enumerate(sub_queries):
                    print(f"{i+1}. {query}")
                
                queries_to_search = [topic] + sub_queries
            
            # 3. Retrieve Documents from MongoDB for all queries
            print(f"\nRetrieving relevant documents for the topic and sub-queries...")
            all_unique_docs = self.retrieve_and_deduplicate_docs(queries_to_search)
            
            if not all_unique_docs:
                logging.warning(f"No documents found in MongoDB for topic '{topic}' or its sub-queries.")
                print("No relevant documents found in the database for this topic.")
                context_str = "No relevant information found in the database."
            else:
                print(f"Retrieved {len(all_unique_docs)} unique relevant documents.")
                
                # 4. Format Context from retrieved documents
                context_str = self.formatter.format(all_unique_docs)
            
            # 5. Generate Essay using Gemini
            print("\nGenerating UPSC essay using Gemini...")
            print("(This may take a minute or two depending on essay complexity)")
            
            essay_content = self.gemini_client.generate_essay(topic, context_str)
            
            # 6. Save Essay as PDF
            if essay_content:
                word_count = len(essay_content.split())
                print(f"\nGenerated essay with approximately {word_count} words.")
                
                print("\n------ ESSAY PREVIEW (FIRST 500 CHARACTERS) ------")
                print(essay_content[:500] + "...\n")
                print("----------------------------------------------------")
                
                logging.info("Saving generated essay to PDF...")
                pdf_filepath = self.pdf_generator.generate(topic, essay_content)
                
                if pdf_filepath:
                    print(f"\nSuccess! Your UPSC essay has been saved to: {pdf_filepath}")
                else:
                    print("\nFailed to save the essay as a PDF file. Check logs.")
                    
                    # Save as text file as fallback
                    try:
                        safe_topic = re.sub(r'[^\w\-_\. ]', '_', topic)
                        text_filepath = os.path.join(self.config.output_directory, f"UPSC_Essay_{safe_topic}.txt")
                        with open(text_filepath, 'w', encoding='utf-8') as f:
                            f.write(f"UPSC ESSAY: {topic}\n\n")
                            f.write(essay_content)
                        print(f"Saved essay as text file instead: {text_filepath}")
                    except Exception as e:
                        logging.error(f"Failed to save text file: {e}")
            else:
                print("\n[Could not generate an essay for this topic. An error occurred during generation.]")
                logging.warning(f"Failed to generate essay content for topic: {topic}")
            
        except Exception as e:
            logging.exception(f"An unexpected error occurred in UPSCEssayGenerator.run: {e}")
            print("An unexpected error occurred. Check logs for details.")
        finally:
            # 7. Close Connections
            if hasattr(self, 'mongo_querier') and self.mongo_querier:
                self.mongo_querier.close_connection()
            logging.info("Script finished.")
            print(f"\nLog file saved to: {log_filename}")

# --- Main Execution ---
if __name__ == "__main__":
    try:
        app = UPSCEssayGenerator()
        app.run()
    except ValueError as ve:
        logging.critical(f"Application initialization failed: {ve}")
        print(f"Application initialization failed: {ve}")
    except Exception as e:
        logging.critical(f"Failed to create UPSCEssayGenerator: {e}", exc_info=True)
        print(f"A critical error occurred during setup: {e}")