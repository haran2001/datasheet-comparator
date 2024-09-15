import os
import re
import openai
import tiktoken
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
from dotenv import load_dotenv
import docx

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key')  # Replace with a secure key in production
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'txt'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize tokenizer for accurate token counting
ENCODER = tiktoken.get_encoding("gpt2")  # GPT-3 uses 'gpt2' encoding

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text(file_path):
    """Extract text from PDF, DOCX, or TXT files."""
    ext = file_path.rsplit('.', 1)[1].lower()
    text = ""
    if ext == 'pdf':
        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            text = f"Error reading PDF: {str(e)}"
    elif ext == 'txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            text = f"Error reading TXT: {str(e)}"
    elif ext == 'docx':
        try:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + '\n'
        except Exception as e:
            text = f"Error reading DOCX: {str(e)}"
    else:
        text = "Unsupported file format."
    return text

def preprocess_text(text):
    """
    Preprocess the extracted text to remove unnecessary tokens.
    """
    # Convert to lowercase for uniformity
    text = text.lower()

    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)

    # Remove headers and footers (assuming they repeat every page)
    # Example pattern: company name or disclaimer at the start/end
    # This needs to be adjusted based on actual datasheet content
    text = re.sub(r'(company a|company b).+?\. ', '', text)

    # Remove table of contents
    text = re.sub(r'table of contents.+?(?=introduction|executive summary|1\.)', '', text)

    # Remove disclaimers and legal texts
    text = re.sub(r'disclaimer.+?(?=introduction|executive summary|1\.)', '', text)
    text = re.sub(r'warranty.+?(?=introduction|executive summary|1\.)', '', text)
    text = re.sub(r'liability.+?(?=introduction|executive summary|1\.)', '', text)

    # Remove page numbers (e.g., "Page 1 of 10")
    text = re.sub(r'page \d+ of \d+', '', text)

    # Remove repeated sections or glossary
    text = re.sub(r'glossary.+', '', text)

    # Remove any remaining non-alphanumeric characters except for basic punctuation
    text = re.sub(r'[^a-zA-Z0-9.,;:()\-–—]', ' ', text)

    # Remove extra spaces again after cleaning
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def count_tokens(text):
    """
    Count the number of tokens in the text using tiktoken.
    """
    tokens = ENCODER.encode(text)
    return len(tokens)

def truncate_text(text, max_tokens=10000):
    """
    Truncate the text to fit within the max_tokens limit.
    """
    tokens = ENCODER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return ENCODER.decode(truncated_tokens)

def compare_datasheets(text_a, text_b):
    """
    Use OpenAI's ChatCompletion to compare two datasheets and generate an analysis.
    """
    prompt = f"""
    You are an expert semiconductor product analyst. Compare the following two product datasheets and provide a detailed analysis of the differences. Highlight strengths and weaknesses of each product and suggest improvements for our product. Conclude with a strategic roadmap to enhance our product based on the comparison.

    **Datasheet 1 (Company A):**
    {text_a}

    **Datasheet 2 (Company B):**
    {text_b}

    **Analysis:**
    """

    # Ensure prompt doesn't exceed token limits
    total_tokens = count_tokens(prompt)
    max_tokens = 4096  # GPT-3's max tokens for a single request; adjust as needed
    if total_tokens > max_tokens:
        # Truncate the input texts proportionally
        available_tokens = max_tokens - count_tokens("**Analysis:**\n")
        per_text_tokens = available_tokens // 2
        text_a = truncate_text(text_a, max_tokens=per_text_tokens)
        text_b = truncate_text(text_b, max_tokens=per_text_tokens)
        prompt = f"""
    You are an expert semiconductor product analyst tasked with comparing two product datasheets from Company A and Company B. Your goal is to provide a comprehensive analysis that highlights the differences, strengths, and weaknesses of each product. Additionally, you should suggest actionable improvements for our product and outline a strategic roadmap based on your findings.

**Datasheet 1 (Company A):**


**Datasheet 2 (Company B):**

**Instructions:**

1. **Executive Summary**
   - Provide a brief overview summarizing the key differences and similarities between the two products.

2. **Detailed Comparison**
   For each of the following critical aspects, compare the two products, highlighting strengths and weaknesses:

   a. **Performance Metrics**
      - Clock Speed
      - Processing Power
      - Efficiency
      - Benchmark Results

   b. **Power Consumption**
      - Voltage Requirements
      - Power Efficiency
      - Thermal Design Power (TDP)

   c. **Thermal Management**
      - Cooling Solutions
      - Heat Dissipation Techniques
      - Operating Temperature Range

   d. **Integration and Compatibility**
      - Supported Interfaces (e.g., PCIe, USB)
      - Compatibility with Existing Systems
      - Ease of Integration

   e. **Cost Analysis**
      - Production Costs
      - Pricing Strategy
      - Cost-Effectiveness

   f. **Reliability and Warranty**
      - Mean Time Between Failures (MTBF)
      - Warranty Period
      - Reliability Ratings

   g. **Manufacturing Process**
      - Fabrication Technology (e.g., 7nm, 10nm)
      - Yield Rates
      - Scalability of Production

   h. **Environmental Impact**
      - Energy Consumption
      - Use of Sustainable Materials
      - Compliance with Environmental Standards

   i. **Packaging and Form Factor**
      - Physical Dimensions
      - Packaging Materials
      - Mounting Options

   j. **Availability and Support**
      - Supply Chain Reliability
      - Customer Support Services
      - Documentation and Resources

   k. **Features and Specifications**
      - Unique Selling Points (USPs)
      - Technical Specifications
      - Optional Features

   l. **Security Features**
      - Built-in Security Mechanisms
      - Compliance with Security Standards
      - Vulnerability Protections

   m. **Scalability**
      - Ability to Scale Performance
      - Support for Future Upgrades
      - Modular Design Elements

3. **Strengths and Weaknesses**
   - Summarize the primary strengths and weaknesses of each product based on the detailed comparison.

4. **Recommendations for Improvement**
   - Provide actionable suggestions to enhance our product by leveraging the strengths observed and addressing the weaknesses identified in the comparison.

5. **Strategic Roadmap**
   - Outline a strategic roadmap that includes short-term and long-term initiatives to implement the recommended improvements. Structure the roadmap with clear timelines and milestones. For example:

     - **Q1 2024:**
       - Initiate R&D on power-efficient architectures.
       - Begin design modifications for improved thermal management.

     - **Q2 2024:**
       - Prototype new power-efficient models.
       - Test and validate thermal solutions in various environments.

     - **Q3 2024:**
       - Initiate partnerships for broader compatibility standards.
       - Begin cost analysis for production optimization.

     - **Q4 2024:**
       - Finalize design for next-gen products incorporating improvements.
       - Plan marketing strategy highlighting enhanced features and competitive pricing.

     - **Q1 2025:**
       - Launch updated product line.
       - Monitor market feedback and iterate as necessary.

**Formatting Requirements:**
- Use clear and descriptive headings for each section.
- Present comparisons in a structured format, such as bullet points or tables, for readability.
- Ensure the analysis is objective, data-driven, and free from biases.
- Maintain professional and concise language throughout the report.

**Additional Notes:**
- If specific data is missing from either datasheet, note the absence and consider its potential impact on the analysis.
- Highlight any innovative technologies or features that set one product apart from the other.
- Consider the target market and use-cases for each product when evaluating their strengths and weaknesses.

---

**Example Output Structure:**


    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You analyze semiconductor product datasheets."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,  # Adjust based on desired response length
            temperature=0.2,
        )
        analysis = response['choices'][0]['message']['content'].strip()
    except Exception as e:
        analysis = f"An error occurred during analysis: {str(e)}"

    return analysis

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if both files are in the request
        if 'datasheet_a' not in request.files or 'datasheet_b' not in request.files:
            flash('Please upload both datasheets.')
            return redirect(request.url)

        file_a = request.files['datasheet_a']
        file_b = request.files['datasheet_b']

        # Validate that files have been selected
        if file_a.filename == '' or file_b.filename == '':
            flash('No selected file.')
            return redirect(request.url)
        if not allowed_file(file_a.filename) or not allowed_file(file_b.filename):
            flash('Unsupported file format. Allowed formats: PDF, DOCX, TXT.')
            return redirect(request.url)

        # Securely save the uploaded files
        filename_a = secure_filename(file_a.filename)
        filename_b = secure_filename(file_b.filename)
        path_a = os.path.join(app.config['UPLOAD_FOLDER'], filename_a)
        path_b = os.path.join(app.config['UPLOAD_FOLDER'], filename_b)
        file_a.save(path_a)
        file_b.save(path_b)

        # Extract text from the uploaded files
        text_a = extract_text(path_a)
        text_b = extract_text(path_b)

        # Preprocess the extracted text
        cleaned_text_a = preprocess_text(text_a)
        cleaned_text_b = preprocess_text(text_b)

        # Ensure the token limit is not exceeded
        cleaned_text_a = truncate_text(cleaned_text_a, max_tokens=10000)
        cleaned_text_b = truncate_text(cleaned_text_b, max_tokens=10000)

        # Use the LLM to compare datasheets
        analysis = compare_datasheets(cleaned_text_a, cleaned_text_b)

        # Optionally, remove the uploaded files after processing
        try:
            os.remove(path_a)
            os.remove(path_b)
        except Exception as e:
            print(f"Error deleting files: {str(e)}")

        return render_template('result.html', analysis=analysis)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
