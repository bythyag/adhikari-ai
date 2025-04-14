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
        self.max_output_tokens: int = 10000 # Increase if needed, check model limits
        self.max_output_tokens_section: int = 2000 # Example: Dedicated limit for section generation
        self.gemini_temperature: float = 0.3 # Controls creativity vs factualness
        self.gemini_temperature_plan: float = 0.2 # More deterministic for planning
        self.gemini_temperature_section: float = 0.4 # Slightly more creative for section writing
        self.output_directory: str = "short_notes"
        self.doc_text_field: str = 'text' # Field containing document text in MongoDB
        self.doc_id_field: str = '_id' # Field containing the unique document ID in MongoDB
        # --- Prompt File Paths ---
        self.prompts_dir: str = "adhikari-ai/prompts"
        self.research_plan_prompt_file: str = os.path.join(self.prompts_dir, "research_plan_prompt.txt") # New
        self.section_content_prompt_file: str = os.path.join(self.prompts_dir, "section_content_prompt.txt") # New
        self.sub_query_prompt_file: str = os.path.join(self.prompts_dir, "short_note_sub_query_prompt.txt")
        self.short_note_prompt_file: str = os.path.join(self.prompts_dir, "short_note_prompt.txt")


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
        # Validate prompt files exist
        if not os.path.exists(self.sub_query_prompt_file):
            logging.error(f"Sub-query prompt file not found: {self.sub_query_prompt_file}")
            print(f"Error: Sub-query prompt file not found at {self.sub_query_prompt_file}")
            return False
        if not os.path.exists(self.short_note_prompt_file):
            logging.error(f"Short note prompt file not found: {self.short_note_prompt_file}")
            print(f"Error: Short note prompt file not found at {self.short_note_prompt_file}")
            return False
        if not os.path.exists(self.research_plan_prompt_file): # New check
            logging.error(f"Research plan prompt file not found: {self.research_plan_prompt_file}")
            print(f"Error: Research plan prompt file not found at {self.research_plan_prompt_file}")
            return False
        if not os.path.exists(self.section_content_prompt_file): # New check
            logging.error(f"Section content prompt file not found: {self.section_content_prompt_file}")
            print(f"Error: Section content prompt file not found at {self.section_content_prompt_file}")
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
        # Config for general/sub-query generation (can be reused or specialized)
        self.generation_config = types.GenerationConfig(
            temperature=config.gemini_temperature,
            max_output_tokens=config.max_output_tokens # General purpose limit
        )
        # Specific config for plan generation (more deterministic)
        self.plan_generation_config = types.GenerationConfig(
            temperature=config.gemini_temperature_plan,
            max_output_tokens=500 # Plan should be relatively short
        )
        # Specific config for section content generation (potentially higher token limit)
        self.section_generation_config = types.GenerationConfig(
            temperature=config.gemini_temperature_section,
            max_output_tokens=config.max_output_tokens_section # Dedicated limit
        )
        self.sub_query_generation_config = types.GenerationConfig( # Keep for sub-queries
            temperature=0.5,
            max_output_tokens=500
        )

        # Load prompt templates
        try:
            self.research_plan_prompt_template = self._load_prompt_template(config.research_plan_prompt_file) # New
            self.section_content_prompt_template = self._load_prompt_template(config.section_content_prompt_file) # New
            self.sub_query_prompt_template = self._load_prompt_template(config.sub_query_prompt_file)
            # self.short_note_prompt_template = self._load_prompt_template(config.short_note_prompt_file) # Remove if not used
            logging.info("Successfully loaded prompt templates.")
        except FileNotFoundError as e:
            logging.error(f"Error loading prompt file: {e}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred loading prompt files: {e}", exc_info=True)
            raise

        logging.info(f"Gemini client configured for model '{config.gemini_model_name}'.")

    def _load_prompt_template(self, filepath: str) -> str:
        """Loads a prompt template from a file."""
        try:
            with open(filepath, 'r') as f:
                return f.read()
        except FileNotFoundError:
            logging.error(f"Prompt file not found: {filepath}")
            raise
        except Exception as e:
            logging.error(f"Error reading prompt file {filepath}: {e}", exc_info=True)
            raise

    def generate_research_plan(self, topic: str) -> Optional[List[str]]:
        """Generates a research plan (list of section titles) for the topic."""
        try:
            prompt = self.research_plan_prompt_template.format(topic=topic)
        except KeyError as e:
             logging.error(f"Missing placeholder in research_plan_prompt.txt: {e}")
             print(f"Error: Placeholder {e} missing in research_plan_prompt.txt.")
             return None

        try:
            model = genai.GenerativeModel(
                self.config.gemini_model_name,
                generation_config=self.plan_generation_config # Use plan config
                # No system instruction needed if prompt is detailed enough
            )
            logging.info(f"Generating research plan for topic: {topic}")
            response = model.generate_content(prompt)

            # Add response validation (check for blocks, empty parts etc.) - similar to generate_sub_queries
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                 logging.warning(f"Gemini Research Plan Prompt Feedback: {response.prompt_feedback}")
            if not response.parts:
                 # Handle blocked/failed generation (similar to other methods)
                 logging.error("Research plan generation blocked or failed.")
                 print("Research plan generation failed or was blocked. Check logs.")
                 return None

            # Attempt to parse the response as JSON list
            try:
                cleaned_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
                plan = json.loads(cleaned_text)
                if isinstance(plan, list) and all(isinstance(section, str) for section in plan):
                    logging.info(f"Successfully generated research plan with {len(plan)} sections.")
                    return plan
                else:
                    logging.error(f"Gemini returned research plan in unexpected format: {plan}")
                    print("Failed to parse research plan from Gemini response. Check logs.")
                    return None
            except json.JSONDecodeError as jde:
                logging.error(f"Failed to decode JSON response for research plan: {jde}. Response text: {response.text}")
                print("Failed to parse research plan (JSON decode error). Check logs.")
                # Fallback: Try splitting by newline if JSON fails? Or just fail.
                # lines = response.text.strip().split('\n')
                # plan = [line.strip('- ').strip() for line in lines if line.strip()]
                # if plan:
                #     logging.warning("Failed JSON parsing for plan, using newline splitting as fallback.")
                #     return plan
                return None
            except Exception as e:
                 logging.error(f"Error processing research plan response: {e}. Response text: {response.text}", exc_info=True)
                 print("An error occurred while processing the research plan. Check logs.")
                 return None

        # Add exception handling (BlockedPromptException, StopCandidateException, etc.) - similar to other methods
        except Exception as e:
            logging.error(f"Error generating research plan with Gemini: {e}", exc_info=True)
            print("An error occurred during research plan generation. Check logs.")
            return None

    def generate_section_content(self, main_topic: str, section_title: str, context: str) -> Optional[str]:
        """Generates detailed content for a specific section using context."""
        try:
            prompt = self.section_content_prompt_template.format(
                main_topic=main_topic,
                section_title=section_title,
                context=context
            )
        except KeyError as e:
             logging.error(f"Missing placeholder in section_content_prompt.txt: {e}")
             print(f"Error: Placeholder {e} missing in section_content_prompt.txt.")
             return None

        try:
            model = genai.GenerativeModel(
                self.config.gemini_model_name,
                generation_config=self.section_generation_config # Use section config
            )
            logging.info(f"Generating content for section: '{section_title}' (Topic: {main_topic})")
            response = model.generate_content(prompt)

            # Add response validation (check for blocks, empty parts etc.) - similar to generate_short_note
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                 logging.warning(f"Gemini Section Content Prompt Feedback: {response.prompt_feedback}")
            if not response.parts:
                 # Handle blocked/failed generation
                 logging.error(f"Content generation failed/blocked for section: {section_title}")
                 print(f"Content generation failed or was blocked for section '{section_title}'. Check logs.")
                 return None

            logging.info(f"Successfully generated content for section: '{section_title}'")
            return response.text

        # Add exception handling (BlockedPromptException, StopCandidateException, etc.) - similar to generate_short_note
        except Exception as e:
            logging.error(f"Error generating content for section '{section_title}': {e}", exc_info=True)
            print(f"An error occurred during content generation for section '{section_title}'. Check logs.")
            return None

    # Keep generate_sub_queries as it might be used per section
    def generate_sub_queries(self, topic: str) -> Optional[List[str]]:
        # ... (implementation remains the same, but it will be called with section titles) ...
        # Format the system instruction using the loaded template
        try:
            system_instruction = self.sub_query_prompt_template.format(
                num_sub_queries=self.config.num_sub_queries,
                topic=topic # Use the passed topic (which might be a section title)
            )
        except KeyError as e:
             logging.error(f"Missing placeholder in sub_query_prompt.txt: {e}")
             print(f"Error: Placeholder {e} missing in sub_query_prompt.txt. Check the file.")
             return None

        prompt = f"Generate {self.config.num_sub_queries} specific sub-queries for the research section: {topic}" # Modified prompt

        try:
            model = genai.GenerativeModel(
                self.config.gemini_model_name,
                generation_config=self.sub_query_generation_config, # Use specific config
                system_instruction=system_instruction
            )
            logging.info(f"Generating sub-queries for section/topic: {topic}")
            response = model.generate_content(prompt)

            # ... rest of the sub-query generation logic remains the same ...
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
    """Orchestrates the deep research document generation process."""
    def __init__(self):
        logging.info("Initializing Deep Research Generator Application...")
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
        logging.info("Deep Research Generator Application initialized successfully.")

    def run(self):
        """Executes the deep research generation workflow."""
        final_content = ""
        try:
            # 1. Get Topic from User
            main_topic = input("Enter the main topic for the deep research document: ")
            if not main_topic:
                logging.warning("No topic entered. Exiting.")
                print("No topic entered.")
                return

            # 2. Generate Research Plan
            print(f"Generating research plan for: {main_topic}...")
            research_plan = self.gemini_client.generate_research_plan(main_topic)

            if not research_plan:
                logging.error(f"Failed to generate research plan for topic: '{main_topic}'. Aborting.")
                print("Error: Could not generate a research plan for this topic. Check logs.")
                return

            logging.info(f"Generated research plan: {research_plan}")
            print(f"Research plan generated with {len(research_plan)} sections.")

            # 3. Iterate through sections, generate content for each
            all_section_content = []
            total_sections = len(research_plan)
            for i, section_title in enumerate(research_plan):
                print(f"\n--- Processing Section {i+1}/{total_sections}: {section_title} ---")
                logging.info(f"--- Processing Section {i+1}/{total_sections}: {section_title} ---")

                # 3a. Generate Sub-queries for the section
                print(f"Generating sub-queries for section: '{section_title}'...")
                sub_queries = self.gemini_client.generate_sub_queries(section_title) # Use section title as topic
                if not sub_queries:
                    logging.warning(f"Could not generate sub-queries for section: '{section_title}'. Proceeding with section title only.")
                    print("Warning: Could not generate sub-queries. Using section title for search.")
                    queries_to_search = [section_title]
                else:
                    logging.info(f"Generated sub-queries for section: {sub_queries}")
                    print(f"Generated {len(sub_queries)} sub-queries for this section.")
                    queries_to_search = [section_title] + sub_queries # Search section title + sub-queries

                # 3b. Retrieve Documents for the section
                print(f"Retrieving documents relevant to '{section_title}'...")
                section_docs_retrieved = []
                # seen_doc_ids_section = set() # Track IDs *per section* if needed, or use global deduplication later

                for q_idx, query in enumerate(queries_to_search):
                    logging.info(f"Searching documents for section query {q_idx+1}/{len(queries_to_search)}: '{query}'")
                    try:
                        search_results = self.mongo_querier.search_similar_documents(
                            query,
                            top_k=self.config.num_retrieved_docs_per_query # Fetch enough docs per query
                        )
                        if search_results:
                            section_docs_retrieved.extend(search_results)
                            logging.info(f"Added {len(search_results)} documents from query '{query}'. Total for section: {len(section_docs_retrieved)}")
                        elif search_results is None:
                             logging.error(f"Error retrieving documents for section query: '{query}'")
                             print(f"Warning: Failed to retrieve documents for query '{query}'. Check logs.")
                        else:
                            logging.info(f"No documents found for section query: '{query}'")
                    except Exception as e:
                        logging.exception(f"Error during document search for section query '{query}': {e}")
                        print(f"An error occurred while searching for query '{query}'. Check logs.")

                logging.info(f"Total documents retrieved for section '{section_title}': {len(section_docs_retrieved)}")

                # Deduplicate documents retrieved *for this section*
                seen_doc_ids_section = set()
                unique_docs_section = []
                for doc in section_docs_retrieved:
                    doc_id = doc.get(self.config.doc_id_field)
                    if doc_id is None: continue
                    doc_id_str = str(doc_id)
                    if doc_id_str not in seen_doc_ids_section:
                        seen_doc_ids_section.add(doc_id_str)
                        unique_docs_section.append(doc)
                
                logging.info(f"Deduplicated to {len(unique_docs_section)} unique documents for section '{section_title}'.")
                print(f"Found {len(unique_docs_section)} unique relevant documents for this section.")

                if not unique_docs_section:
                    logging.warning(f"No unique documents found for section '{section_title}'. Skipping content generation for this section.")
                    print("Warning: No relevant documents found for this section. Content generation skipped.")
                    all_section_content.append(f"# {section_title}\n\n[No relevant information found in the database for this section.]\n\n")
                    continue # Skip to the next section

                # 3c. Format Context for the section
                context_str = self.formatter.format(unique_docs_section)

                # 3d. Generate Content for the section using Gemini
                print(f"Generating content for section: '{section_title}'...")
                logging.info(f"Generating content for section '{section_title}' using {len(unique_docs_section)} documents context.")
                section_content = self.gemini_client.generate_section_content(main_topic, section_title, context_str)

                if section_content:
                    # Add section title as a markdown heading before the content
                    formatted_section = f"# {section_title}\n\n{section_content.strip()}\n\n"
                    all_section_content.append(formatted_section)
                    print(f"Successfully generated content for section: '{section_title}'.")
                    logging.info(f"Successfully generated content for section: '{section_title}'. Word count approx: {len(section_content.split())}")
                else:
                    logging.warning(f"Failed to generate content for section: '{section_title}'.")
                    print(f"Warning: Failed to generate content for section '{section_title}'. Adding placeholder.")
                    # Add placeholder if generation fails but context was present
                    all_section_content.append(f"# {section_title}\n\n[Error generating content for this section. Context was available.]\n\n")

            # 4. Combine all generated sections
            print("\n--- Combining all sections ---")
            final_content = "".join(all_section_content)

            if not final_content.strip():
                 logging.error("Failed to generate content for any section. Final document is empty.")
                 print("\nError: No content could be generated for any section. Final document is empty.")
                 return

            # Estimate final word count
            final_word_count = len(final_content.split())
            logging.info(f"Combined final document content. Estimated word count: {final_word_count}")
            print(f"Combined document generated with estimated {final_word_count} words.")

            # 5. Save Final Document as PDF
            print("Saving final document to PDF...")
            logging.info("Saving final document to PDF...")
            # Use the main topic for the filename
            pdf_filepath = self.pdf_generator.generate(main_topic, final_content)

            if pdf_filepath:
                print(f"\nDeep research document saved successfully to: {pdf_filepath}")
            else:
                print("\nError: Failed to save the final document as a PDF file. Check logs.")
                # Optionally print the final content to console if PDF fails
                # print("\n--- Final Document Content ---")
                # print(final_content)
                # print("-----------------------------")


        except Exception as e:
            logging.exception(f"An unexpected error occurred in ShortNotesGeneratorApp.run: {e}")
            print(f"An unexpected error occurred: {e}. Check logs for details.")
        finally:
            # 6. Close Connections
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
