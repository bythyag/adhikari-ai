import os
import re
import datetime
import html
import json
import time
from tavily import TavilyClient
from dotenv import load_dotenv
import logging
import google.generativeai as genai
from google.generativeai import types

# --- PDF Generation (Adapted from short_notes_generator.py) ---
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
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
log_filename = os.path.join(log_directory, f"daily_news_gemini_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_filename,
    filemode='w'
)

# --- Configuration ---
class NewsConfig:
    """Loads and stores configuration parameters."""
    def __init__(self):
        load_dotenv()
        self.tavily_api_key: str | None = os.getenv("TAVILY_API_KEY")
        self.gemini_api_key: str | None = os.getenv("GEMINI_API_KEY") # Added
        self.output_directory: str = "daily_news_reports_gemini"
        # UPSC Topics to search for
        self.broad_topics: list[str] = [ # Renamed for clarity
            "India Polity and Governance",
            "India Economy and Development",
            "India Geography",
            "India Environment and Ecology",
            "India Science and Technology",
            "India Art, Culture, and History context in news",
            "India International Relations and Security",
            "Social Issues in India",
            "Current Events of National and International Importance"
        ]
        self.num_queries_per_topic: int = 3 # How many specific queries Gemini should generate per broad topic
        self.max_results_per_query: int = 3 # Number of results Tavily fetches per specific query
        self.search_depth: str = "basic" # or "advanced"
        self.gemini_model_name: str = "models/gemini-2.0-flash"
        self.gemini_temperature_query: float = 0.5 # For query generation
        self.gemini_temperature_report: float = 0.3 # For report compilation (more factual)
        self.max_output_tokens_report: int = 4096 # Max tokens for the final report
        self.max_context_tokens_report: int = 15000 # Approx limit for context to Gemini (adjust based on model)

        # --- Prompt File Paths ---
        self.prompts_dir: str = "prompts"
        self.query_generation_prompt_file: str = os.path.join(self.prompts_dir, "query_generation_prompt.txt")
        self.report_compilation_prompt_file: str = os.path.join(self.prompts_dir, "report_compilation_prompt.txt")


    def validate(self) -> bool:
        """Basic validation of essential configurations."""
        if not self.tavily_api_key:
            logging.error("TAVILY_API_KEY environment variable not set.")
            print("Error: TAVILY_API_KEY environment variable not set.")
            return False
        if not self.gemini_api_key: # Added
            logging.error("GEMINI_API_KEY environment variable not set.")
            print("Error: GEMINI_API_KEY environment variable not set.")
            return False
        if not os.path.exists(self.query_generation_prompt_file):
            logging.error(f"Query generation prompt file not found: {self.query_generation_prompt_file}")
            print(f"Error: Query generation prompt file not found at {self.query_generation_prompt_file}")
            return False
        if not os.path.exists(self.report_compilation_prompt_file):
            logging.error(f"Report compilation prompt file not found: {self.report_compilation_prompt_file}")
            print(f"Error: Report compilation prompt file not found at {self.report_compilation_prompt_file}")
            return False
        if not REPORTLAB_AVAILABLE:
            logging.warning("reportlab library is required for PDF generation but not found.")
            # Allow script to run but log warning
        return True

# --- Gemini Client ---
class GeminiClient:
    """Handles interactions with the Google Gemini API."""
    def __init__(self, config: NewsConfig):
        self.config = config
        try:
            genai.configure(api_key=config.gemini_api_key)
            logging.info("Gemini API configured successfully.")
        except Exception as e:
            logging.error(f"Failed to configure Gemini API: {e}", exc_info=True)
            raise ValueError("Failed to configure Gemini API. Check API key and permissions.") from e

        # Generation configs
        self.query_gen_config = types.GenerationConfig(
            temperature=config.gemini_temperature_query,
            response_mime_type="application/json", # Expect JSON for queries
        )
        self.report_gen_config = types.GenerationConfig(
            temperature=config.gemini_temperature_report,
            max_output_tokens=config.max_output_tokens_report,
            # response_mime_type="text/plain", # Default
        )

        # Load prompt templates
        try:
            self.query_gen_prompt_template = self._load_prompt_template(config.query_generation_prompt_file)
            self.report_gen_prompt_template = self._load_prompt_template(config.report_compilation_prompt_file)
            logging.info("Successfully loaded Gemini prompt templates.")
        except FileNotFoundError as e:
            logging.error(f"Error loading prompt file: {e}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred loading prompt files: {e}", exc_info=True)
            raise

        self.model_query = genai.GenerativeModel(config.gemini_model_name)
        self.model_report = genai.GenerativeModel(config.gemini_model_name)
        logging.info(f"Gemini client initialized with model '{config.gemini_model_name}'.")

    def _load_prompt_template(self, filepath: str) -> str:
        """Loads a prompt template from a file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logging.error(f"Prompt file not found: {filepath}")
            raise
        except Exception as e:
            logging.error(f"Error reading prompt file {filepath}: {e}", exc_info=True)
            raise

    def _call_gemini_with_retry(self, model, prompt, generation_config, max_retries=3, delay=5):
        """Calls Gemini API with retry logic for potential rate limits or transient errors."""
        retries = 0
        while retries < max_retries:
            try:
                response = model.generate_content(prompt, generation_config=generation_config)
                # Basic validation
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    block_reason = response.prompt_feedback.block_reason
                    logging.error(f"Gemini prompt blocked. Reason: {block_reason}")
                    print(f"Warning: Gemini request blocked due to safety settings ({block_reason}). Skipping.")
                    return None # Indicate blocked prompt

                if not response.parts:
                    logging.warning("Gemini response has no parts (empty response).")
                    # Treat as empty/failed for query gen, maybe retry for report gen?
                    # For now, return None for both.
                    return None

                return response # Success

            except types.generation_types.BlockedPromptException as bpe:
                 logging.error(f"Gemini prompt blocked by API: {bpe}")
                 print("Warning: Gemini request blocked by API safety filter. Skipping.")
                 return None # Blocked, don't retry
            except types.generation_types.StopCandidateException as sce:
                 logging.error(f"Gemini generation stopped unexpectedly: {sce}")
                 print("Warning: Gemini generation stopped unexpectedly. Skipping.")
                 return None # Stopped, don't retry
            except Exception as e:
                # Catch potential API errors (rate limits, server issues)
                logging.warning(f"Error calling Gemini API (Attempt {retries + 1}/{max_retries}): {e}")
                retries += 1
                if retries >= max_retries:
                    logging.error("Max retries reached for Gemini API call.")
                    print("Error: Failed to get response from Gemini after multiple retries. Check logs.")
                    return None
                print(f"Retrying Gemini call in {delay} seconds...")
                time.sleep(delay)
        return None # Should not be reached if max_retries > 0

    def generate_search_queries(self, topic: str, date: str) -> list[str] | None:
        """Generates specific search queries for a broad topic using Gemini."""
        try:
            prompt = self.query_gen_prompt_template.format(
                num_queries=self.config.num_queries_per_topic,
                topic=topic,
                date=date
            )
        except KeyError as e:
             logging.error(f"Missing placeholder in query_generation_prompt.txt: {e}")
             print(f"Error: Placeholder {e} missing in query_generation_prompt.txt.")
             return None

        logging.info(f"Generating search queries for topic: {topic}")
        response = self._call_gemini_with_retry(self.model_query, prompt, self.query_gen_config)

        if response is None:
            return None # Error or blocked prompt handled in retry function

        try:
            # Gemini should return JSON directly due to response_mime_type
            query_list = json.loads(response.text)
            if isinstance(query_list, list) and all(isinstance(q, str) for q in query_list):
                logging.info(f"Successfully generated {len(query_list)} queries for '{topic}'.")
                return query_list
            else:
                logging.error(f"Gemini returned unexpected format for queries (expected list of strings): {response.text}")
                print(f"Warning: Could not parse queries for '{topic}' from Gemini response.")
                return None
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON response from Gemini for queries: {response.text}")
            print(f"Warning: Could not parse queries for '{topic}' from Gemini response.")
            return None
        except Exception as e:
            logging.error(f"Error processing Gemini query response for '{topic}': {e}", exc_info=True)
            print(f"Warning: Error processing generated queries for '{topic}'.")
            return None

    def compile_report(self, context: str, date: str) -> str | None:
        """Compiles the final news report using Gemini."""
        # Basic check for context size (very approximate token count)
        if len(context.split()) > self.config.max_context_tokens_report * 0.8: # Use 80% as a buffer
             logging.warning(f"Context length ({len(context.split())} words) might exceed model limits. Truncating is not implemented, Gemini might fail.")
             print("Warning: Large amount of news data collected, might exceed LLM context limit.")

        try:
            prompt = self.report_gen_prompt_template.format(
                date=date,
                context=context
            )
        except KeyError as e:
             logging.error(f"Missing placeholder in report_compilation_prompt.txt: {e}")
             print(f"Error: Placeholder {e} missing in report_compilation_prompt.txt.")
             return None

        logging.info(f"Compiling final report for date: {date}")
        response = self._call_gemini_with_retry(self.model_report, prompt, self.report_gen_config)

        if response is None:
            return None # Error or blocked prompt

        logging.info(f"Successfully generated report content from Gemini.")
        return response.text


# --- PDF Generator (Keep as is, minor adjustments if needed based on Gemini output) ---
class PDFGenerator:
    """Generates a PDF document from text content."""
    def __init__(self, config: NewsConfig):
        self.config = config
        os.makedirs(self.config.output_directory, exist_ok=True)

    def _preprocess_text(self, text: str) -> str:
        """Applies basic XML markup for ReportLab."""
        text = html.escape(text) # Escape HTML special characters FIRST
        # Handle bold (**text**) -> <b>text</b>
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        # Handle italics (*text* or _text_) -> <i>text</i>
        # Handle single asterisk first to avoid conflict with bold
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        text = re.sub(r'_(.*?)_', r'<i>\1</i>', text)
        # Replace escaped entities for bold/italic tags back to symbols AFTER main escape
        text = text.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
        text = text.replace('&lt;i&gt;', '<i>').replace('&lt;/i&gt;', '</i>')
        # Handle newlines within content for better paragraph flow in PDF
        text = text.replace('\n', '<br/>')
        return text

    def generate(self, title: str, content: str) -> str | None:
        """Creates a PDF file with the given title and content."""
        if not REPORTLAB_AVAILABLE:
            logging.error("Cannot generate PDF: reportlab library is not installed.")
            return None

        safe_title = re.sub(r'[^\w\-_\. ]', '_', title)
        filename = os.path.join(self.config.output_directory, f"{safe_title}.pdf")

        try:
            doc = SimpleDocTemplate(filename, pagesize=letter,
                                    leftMargin=inch, rightMargin=inch,
                                    topMargin=inch, bottomMargin=inch)
            styles = getSampleStyleSheet()
            # Customize styles
            styles.add(ParagraphStyle(name='H1', parent=styles['h1'], spaceAfter=0.2*inch, fontSize=16))
            styles.add(ParagraphStyle(name='H2', parent=styles['h2'], spaceAfter=0.15*inch, fontSize=14))
            styles.add(ParagraphStyle(name='H3', parent=styles['h3'], spaceAfter=0.05*inch, spaceBefore=0.1*inch, fontSize=12, textColor='#333333')) # Style for news titles
            styles.add(ParagraphStyle(name='SourceLink', parent=styles['BodyText'], textColor='blue', fontSize=9, spaceBefore=0, spaceAfter=0.1*inch))
            # Style for the analysis/content blockquote
            analysis_style = ParagraphStyle(name='Analysis', parent=styles['BodyText'], leftIndent=0.2*inch, firstLineIndent=0, spaceBefore=0.05*inch, spaceAfter=0.1*inch, leading=14, textColor='#555555')
            body_style = ParagraphStyle(name='BodyTextCustom', parent=styles['BodyText'], leading=14, spaceAfter=0.1*inch)


            story = []
            lines = content.strip().split('\n')

            # Find the main title (first non-empty line, assumed to be H1)
            main_title_text = "UPSC Daily News Analysis" # Default
            first_line_processed = False
            processed_lines = []

            for line in lines:
                stripped_line = line.strip()
                if not first_line_processed and stripped_line:
                     # Assume the first non-empty line is the main title generated by Gemini
                     # Check if it looks like a markdown H1, otherwise use default
                     if stripped_line.startswith('# '):
                         main_title_text = stripped_line[2:].strip()
                     else:
                         main_title_text = stripped_line # Use the line as is if not H1 markdown
                         processed_lines.append(line) # Keep it for body processing if not H1
                     first_line_processed = True
                elif first_line_processed:
                    processed_lines.append(line) # Add remaining lines

            story.append(Paragraph(html.escape(main_title_text), styles['H1']))
            story.append(Spacer(1, 0.2*inch))


            # Process remaining lines
            for i, line_content in enumerate(processed_lines):
                line = line_content.strip()

                # Skip empty lines or lines that were just the H1 marker
                if not line or line == '#':
                    continue

                try:
                    if line.startswith('## '): # Topic Heading
                        text = self._preprocess_text(line[3:].strip())
                        # Add PageBreak before H2 if it's not the very first element after title
                        if len(story) > 2: # Check if more than Title and Spacer exist
                            story.append(PageBreak())
                        story.append(Paragraph(text, styles['H2']))
                        story.append(Spacer(1, 0.05*inch)) # Reduced space after H2
                    elif line.startswith('### '): # News Item Title
                        text = self._preprocess_text(line[4:].strip())
                        story.append(Paragraph(text, styles['H3']))
                        # No spacer needed after H3, handled by Analysis/Source spaceBefore/After
                    elif line.startswith('Source: '): # News Source Link
                        link = line.split('Source: ', 1)[1].strip()
                        # Basic URL validation (optional but good)
                        if link.startswith('http://') or link.startswith('https://'):
                            text = f'<link href="{html.escape(link)}">{html.escape(link)}</link>'
                            story.append(Paragraph(text, styles['SourceLink']))
                        else:
                            text = self._preprocess_text(f"Source: {link}") # Display as text if not valid link
                            story.append(Paragraph(text, styles['SourceLink']))
                        # story.append(Spacer(1, 0.1*inch)) # Space now handled by H3 spaceAfter or Analysis spaceAfter
                    elif line.startswith('> '): # Analysis/Content Block
                        text = self._preprocess_text(line[2:].strip())
                        story.append(Paragraph(text, analysis_style))
                    else:
                        # Default paragraph for any other text generated by Gemini
                        text = self._preprocess_text(line)
                        story.append(Paragraph(text, body_style))

                except Exception as pe:
                    logging.error(f"PDF Error: Failed processing line {i+1}. Content: '{line_content}'. Error: {pe}", exc_info=True)
                    story.append(Paragraph(f"<i>[Error processing PDF line {i+1}]</i>", body_style))

            logging.info(f"Building PDF document: {filename}")
            doc.build(story)
            logging.info(f"Successfully generated PDF: {filename}")
            return filename

        except Exception as e:
            logging.error(f"Error generating PDF '{filename}': {e}", exc_info=True)
            print(f"Error generating PDF for '{title}'. Check logs.")
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                    logging.info(f"Removed partially created PDF: {filename}")
                except OSError as oe:
                    logging.error(f"Failed to remove partially created PDF '{filename}': {oe}")
            return None


# --- Daily News Analyzer Application ---
class DailyNewsAnalyzerApp:
    """Fetches, analyzes (summarizes), and generates a PDF report of daily news using Tavily and Gemini."""
    def __init__(self):
        logging.info("Initializing Daily News Analyzer Application...")
        self.config = NewsConfig()
        if not self.config.validate():
            raise ValueError("Configuration validation failed. Check logs.")

        self.tavily_client = TavilyClient(api_key=self.config.tavily_api_key)
        self.gemini_client = GeminiClient(self.config) # Added
        self.pdf_generator = PDFGenerator(self.config)
        logging.info("Daily News Analyzer Application initialized successfully.")

    def _get_previous_day_date(self) -> str:
        """Returns the date of the previous day in YYYY-MM-DD format."""
        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        return yesterday.strftime('%Y-%m-%d')

    def fetch_news_with_generated_queries(self) -> dict[str, list[dict]]:
        """Generates queries via Gemini, fetches news via Tavily."""
        previous_day = self._get_previous_day_date()
        all_news_by_topic = {}
        print(f"--- Step 1: Generating Search Queries for {previous_day} ---")
        logging.info(f"Fetching news for date: {previous_day}")

        # Use a set to store unique URLs across all queries to avoid duplicates in context
        seen_urls = set()

        for broad_topic in self.config.broad_topics:
            print(f"  Generating queries for: '{broad_topic}'...")
            specific_queries = self.gemini_client.generate_search_queries(broad_topic, previous_day)

            if not specific_queries:
                logging.warning(f"No specific queries generated for broad topic: {broad_topic}. Skipping.")
                all_news_by_topic[broad_topic] = [] # Ensure key exists
                continue

            print(f"    Generated {len(specific_queries)} queries. Fetching results...")
            topic_results = []
            for query in specific_queries:
                logging.info(f"Searching Tavily for specific query: '{query}' (Broad Topic: {broad_topic})")
                try:
                    # Add date constraint again for safety, though Gemini prompt includes it
                    query_with_date = f"{query} date:{previous_day}"
                    response = self.tavily_client.search(
                        query=query_with_date,
                        search_depth=self.config.search_depth,
                        max_results=self.config.max_results_per_query,
                        include_answer=False
                    )
                    results = response.get('results', [])
                    if results:
                        logging.info(f"    Found {len(results)} results for query: '{query}'")
                        # Filter out duplicates based on URL before adding
                        new_results_count = 0
                        for res in results:
                            url = res.get('url')
                            if url and url not in seen_urls:
                                topic_results.append(res)
                                seen_urls.add(url)
                                new_results_count += 1
                        if new_results_count < len(results):
                             logging.info(f"    Filtered out {len(results) - new_results_count} duplicate URLs for query: '{query}'")

                    else:
                        logging.warning(f"    No results found for query: '{query}'.")

                except Exception as e:
                    logging.error(f"Error fetching news for query '{query}' from Tavily: {e}", exc_info=True)
                    print(f"    Error fetching news for query: '{query}'. Check logs.")
                time.sleep(1) # Small delay between Tavily calls

            if topic_results:
                 logging.info(f"Collected {len(topic_results)} unique results for broad topic: {broad_topic}")
                 all_news_by_topic[broad_topic] = topic_results
            else:
                 logging.warning(f"No unique results collected for broad topic: {broad_topic}")
                 all_news_by_topic[broad_topic] = [] # Ensure key exists

        return all_news_by_topic

    def format_context_for_gemini(self, news_data: dict[str, list[dict]]) -> str:
        """Formats the fetched news data into a context string for the Gemini report compiler."""
        context = ""
        logging.info("Formatting fetched news data into context for Gemini.")

        for topic, results in news_data.items():
            if not results:
                continue

            context += f"Topic: {topic}\n"
            context += "---\n"
            for i, item in enumerate(results):
                title = item.get('title', 'No Title')
                url = item.get('url', '#')
                # Use 'content' from Tavily, often a summary/snippet
                snippet = item.get('content', 'No content available.')
                score = item.get('score', 'N/A') # Include score if available

                context += f"Item {i+1}:\n"
                context += f"  Title: {title}\n"
                context += f"  URL: {url}\n"
                context += f"  Content Snippet: {snippet}\n"
                # context += f"  Relevance Score: {score}\n" # Optional: include score
                context += "\n"
            context += "\n" # Extra newline between topics

        if not context.strip():
            logging.warning("No news content found across all topics to format for context.")
            return "No relevant news found for the specified topics and date."

        logging.info("Successfully formatted news context for Gemini.")
        return context.strip()


    def run(self):
        """Executes the news fetching, analysis, and PDF generation workflow."""
        try:
            # 1. Fetch News using Gemini-generated queries
            news_data = self.fetch_news_with_generated_queries()

            if not any(news_data.values()): # Check if any topic has results
                print("\nNo news articles found for any topic for the previous day after filtering.")
                logging.warning("No news articles found. PDF will not be generated.")
                return

            # 2. Format data as context for Gemini report compilation
            print("\n--- Step 2: Preparing Context for Report Generation ---")
            gemini_context = self.format_context_for_gemini(news_data)

            if gemini_context.startswith("No relevant news"):
                 print("\nNo relevant news context to generate report.")
                 logging.warning("No context available for Gemini report compilation.")
                 return

            # 3. Compile Report using Gemini
            print("\n--- Step 3: Compiling News Analysis Report with Gemini ---")
            previous_day_str = self._get_previous_day_date()
            compiled_report_content = self.gemini_client.compile_report(gemini_context, previous_day_str)

            if not compiled_report_content:
                print("\nError: Failed to generate the final report content using Gemini. Check logs.")
                logging.error("Gemini failed to compile the report.")
                return

            # 4. Generate PDF from Gemini's output
            pdf_title = f"UPSC Daily News Analysis - {previous_day_str}" # Title for the file
            print(f"\n--- Step 4: Generating PDF Report: '{pdf_title}.pdf' ---")
            pdf_filepath = self.pdf_generator.generate(pdf_title, compiled_report_content)

            if pdf_filepath:
                print(f"\nDaily news report saved successfully to: {pdf_filepath}")
            else:
                print("\nError: Failed to save the final document as a PDF file. Check logs.")

        except ValueError as ve:
             # Catch config validation errors during init
             logging.critical(f"Initialization failed: {ve}")
             print(f"Initialization failed: {ve}")
        except Exception as e:
            logging.exception(f"An unexpected error occurred in DailyNewsAnalyzerApp.run: {e}")
            print(f"An unexpected error occurred: {e}. Check logs for details.")
        finally:
            logging.info("Script finished.")
            print(f"\nLog file saved to: {log_filename}")

# --- Main Execution ---
if __name__ == "__main__":
    if not REPORTLAB_AVAILABLE:
         print("Error: reportlab library not found. PDF generation is disabled.")
         print("Please install it using: pip install reportlab")
         # exit() # Uncomment to force exit if PDF is mandatory

    try:
        app = DailyNewsAnalyzerApp()
        app.run()
    except ValueError as ve:
        # Handles initialization errors (e.g., missing config)
        logging.critical(f"Application initialization failed: {ve}")
        print(f"Application initialization failed: {ve}")
    except Exception as e:
        # Catch-all for unexpected errors during instantiation
        logging.critical(f"Failed to create DailyNewsAnalyzerApp: {e}", exc_info=True)
        print(f"An critical error occurred during setup: {e}")
