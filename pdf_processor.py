import fitz  # PyMuPDF
import os
import sys
import traceback

class PDFProcessor:
    """
    A class to handle PDF text extraction and cleaning.
    """

    @staticmethod
    def clean_extracted_text(text_lines):
        """
        Removes the last non-blank line if it's short (< 60 chars) and lacks sentence end punctuation.
        Preserves the first two lines. Also removes trailing blank lines.
        """
        cleaned_lines = list(text_lines) # Mutable copy

        # 1. Remove all trailing blank lines first
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()

        # 2. Check if more than 2 lines remain after removing blanks
        if len(cleaned_lines) > 2:
            last_line = cleaned_lines[-1].strip()
            # 3. Check heuristic: non-empty, short, and no sentence punctuation
            if last_line and len(last_line) < 60 and not last_line.endswith(('.', '?', '!')):
                # print(f"DEBUG: Removing potential caption/footer: '{last_line}'") # Optional debug print
                cleaned_lines.pop() # Remove the last content line

        # 4. Remove trailing blanks again (in case the original had only 1 or 2 lines with blanks)
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()

        return cleaned_lines

    def extract_text_from_pdf(self, pdf_path, output_dir):
        """
        Extracts text from all pages of a PDF file, cleans potential captions/footers,
        and saves each page as a separate text file in the specified output directory.
        """
        # Ensure the input PDF file exists
        if not os.path.exists(pdf_path):
            print(f"Error: Input PDF file not found at '{pdf_path}'")
            return

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get the PDF filename without extension for naming the output files
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

        doc = None # Initialize doc to None
        try:
            # Open the PDF file
            doc = fitz.open(pdf_path)
            print(f"Successfully opened PDF: '{pdf_path}'")
            print(f"Number of pages: {doc.page_count}")

            # Iterate through each page
            for page_num in range(doc.page_count):
                try:
                    page = doc.load_page(page_num)
                    page_text = page.get_text("text")  # Extract text as plain text

                    # Split text into lines and clean
                    lines = page_text.splitlines()
                    cleaned_lines = self.clean_extracted_text(lines) # Use the static method
                    cleaned_text = "\n".join(cleaned_lines)

                    # Create output filename for this page
                    output_file = os.path.join(output_dir, f"{pdf_name}_page_{page_num + 1}.txt")

                    # Write the cleaned page text to a separate file
                    try:
                        with open(output_file, "w", encoding="utf-8") as outfile:
                            # Add a newline at the end only if there's content
                            outfile.write(cleaned_text + ("\n" if cleaned_text else ""))
                        print(f"Processed Page {page_num + 1}/{doc.page_count} -> {output_file}")
                    except IOError as write_err:
                        print(f"Error: Could not write to file '{output_file}'. Error: {write_err}")

                except Exception as page_err:
                    print(f"Warning: Could not extract text from page {page_num + 1}. Error: {page_err}")
                    # Create an error note file for failed pages
                    error_file = os.path.join(output_dir, f"{pdf_name}_page_{page_num + 1}_error.txt")
                    try:
                        with open(error_file, "w", encoding="utf-8") as errfile:
                            errfile.write(f"Could not extract text from page {page_num + 1}\nError: {page_err}")
                    except IOError as err_write_err:
                         print(f"Error: Could not write error file '{error_file}'. Error: {err_write_err}")


        except fitz.fitz.FileNotFoundError:
            print(f"Error: PyMuPDF could not find the file at '{pdf_path}'. Please check the path.")
        except fitz.fitz.PdfCannotOpenError:
            print(f"Error: Could not open or process the PDF file '{pdf_path}'. It might be corrupted, password-protected without providing a password, or not a valid PDF.")
        except Exception as e:
            print(f"An unexpected error occurred during PDF processing for '{pdf_path}': {e}")
            traceback.print_exc()
        finally:
            # Ensure the PDF document is closed even if errors occur
            if doc:
                doc.close()
                print(f"\nClosed PDF: '{pdf_path}'")


    def process_directory(self, input_root_dir, output_base_dir):
        """
        Walks through the input directory, finds all PDFs, and processes them,
        mirroring the directory structure in the output base directory.
        """
        print(f"Starting PDF processing from: {input_root_dir}")
        print(f"Output will be saved to subdirectories within: {output_base_dir}, mirroring the source structure.")

        # Walk through the input directory structure
        for subdir, dirs, files in os.walk(input_root_dir):
            for filename in files:
                if filename.lower().endswith(".pdf"):
                    # Construct the full path to the PDF file
                    pdf_file_path = os.path.join(subdir, filename)

                    # Get the PDF filename without the extension
                    pdf_name_no_ext = os.path.splitext(filename)[0]

                    # Calculate the relative path from the input root to the current subdir
                    relative_subdir = os.path.relpath(subdir, input_root_dir)

                    # Construct the specific output directory path for this PDF
                    # It mirrors the source structure under output_base_dir,
                    # with a final directory named after the PDF file.
                    output_directory = os.path.join(output_base_dir, relative_subdir, pdf_name_no_ext)

                    print(f"\nProcessing PDF: {pdf_file_path}")
                    print(f"Output directory: {output_directory}")

                    # Call the extraction function for the current PDF
                    self.extract_text_from_pdf(pdf_file_path, output_directory)

        print("\n--- All PDF processing complete. ---")


# --- Main execution logic ---
if __name__ == "__main__":
    # Define the root directory containing the NCERT PDFs
    ncert_root_dir = '/Users/thyag/Desktop/projects/upsc/ncert'
    # Define the base directory where cleaned text files will be saved
    cleaned_data_dir = '/Users/thyag/Desktop/projects/cleaned upsc dataset'

    # Create an instance of the processor
    processor = PDFProcessor()

    # Process the directory
    processor.process_directory(ncert_root_dir, cleaned_data_dir)