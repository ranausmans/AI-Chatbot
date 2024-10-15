from flask import Flask, render_template, request, jsonify, session
import os
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from flask_cors import CORS
import PyPDF2
from werkzeug.utils import secure_filename
import uuid
import json
import re

# Configure Google Generative AI
GENAI_API_KEY = "API KEY HERE "
genai.configure(api_key=GENAI_API_KEY)

# Configure Flask app
logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

app.secret_key = 'your_secret_key_here'  # Make sure this is a strong, random key

# Set up the Generative AI model configuration
generation_config = {
    "temperature": 0.15,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 500,
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Knowledge Base Configuration
KNOWLEDGE_BASE_DIR = 'knowledge_base'  # Directory containing knowledge base text files
TOP_N = 3  # Number of top relevant documents to retrieve

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Load and preprocess the knowledge base
documents = []
document_names = []

if not os.path.exists(KNOWLEDGE_BASE_DIR):
    os.makedirs(KNOWLEDGE_BASE_DIR)
    print(f"Created '{KNOWLEDGE_BASE_DIR}' directory. Please add your knowledge base text files there.")
else:
    for filename in os.listdir(KNOWLEDGE_BASE_DIR):
        if filename.endswith('.txt'):
            filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(content)
                document_names.append(filename)
    print(f"Loaded knowledge base from '{KNOWLEDGE_BASE_DIR}' with {len(documents)} documents.")

# Fit the TF-IDF vectorizer on the documents
if documents:
    tfidf_matrix = vectorizer.fit_transform(documents)
else:
    tfidf_matrix = None
    print("No documents found in the knowledge base. Please add .txt files to the 'knowledge_base' directory.")

# Add new configurations
UPLOAD_FOLDER = 'uploads'
USER_DATA_FOLDER = 'user_data'
ALLOWED_EXTENSIONS = {'pdf'}

for folder in [UPLOAD_FOLDER, USER_DATA_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def get_user_folder(user_id):
    user_folder = os.path.join(USER_DATA_FOLDER, user_id)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    return user_folder

def save_user_data(user_id, data):
    user_folder = get_user_folder(user_id)
    with open(os.path.join(user_folder, 'data.json'), 'w') as f:
        json.dump(data, f)

def load_user_data(user_id):
    user_folder = get_user_folder(user_id)
    data_file = os.path.join(user_folder, 'data.json')
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            return json.load(f)
    return {'documents': [], 'document_names': []}

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        user_id = session.get('user_id')
        if not user_id:
            user_id = str(uuid.uuid4())
            session['user_id'] = user_id
        
        filename = secure_filename(file.filename)
        user_folder = get_user_folder(user_id)
        file_path = os.path.join(user_folder, filename)
        file.save(file_path)
        
        # Extract text from PDF and add to documents
        text = extract_text_from_pdf(file_path)
        
        # Load existing user data or create new
        user_data = load_user_data(user_id)
        
        user_data['documents'].append(text)
        user_data['document_names'].append(filename)
        
        # Save updated user data
        save_user_data(user_id, user_data)
        
        return jsonify({'message': 'File uploaded successfully'})
    return jsonify({'error': 'File type not allowed'})

@app.route('/get_files', methods=['GET'])
def get_files():
    user_id = session.get('user_id')
    if user_id:
        user_data = load_user_data(user_id)
        return jsonify({'files': user_data['document_names']})
    return jsonify({'files': []})

@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        user_query = request.form.get("query")
        selected_file = request.form.get("selected_file")
        
        app.logger.info(f"Query: {user_query}, Selected File: {selected_file}")
        
        user_id = session.get('user_id')
        if user_id:
            user_data = load_user_data(user_id)
            if user_data['documents']:
                documents = user_data['documents']
                document_names = user_data['document_names']
                
                app.logger.info(f"User documents: {document_names}")
                
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(documents)
                
                if selected_file and selected_file in document_names:
                    doc_index = document_names.index(selected_file)
                    query_vec = vectorizer.transform([user_query])
                    doc_vec = tfidf_matrix[doc_index]
                    similarities = cosine_similarity(query_vec, doc_vec).flatten()
                    retrieved_content = documents[doc_index]
                    app.logger.info(f"Using selected file: {selected_file}")
                else:
                    query_vec = vectorizer.transform([user_query])
                    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
                    top_indices = similarities.argsort()[-TOP_N:][::-1]
                    retrieved_docs = [documents[idx] for idx in top_indices]
                    retrieved_content = "\n\n".join(retrieved_docs)
                    app.logger.info("Using all documents")

                prompt = f"Context:\n{retrieved_content}\n\nQuestion: {user_query}\nAnswer:"

                try:
                    api_response = model.generate_content(prompt)
                    response = format_response(api_response.text.strip())
                except Exception as e:
                    response = f"An error occurred while generating the response: {e}"
                    app.logger.error(f"Error generating response: {e}")
            else:
                response = "No documents have been uploaded yet. Please upload a PDF file first."
        else:
            response = "Please upload a document first."

    return render_template("index.html", response=response)

def format_response(response_text):
    # Split the response into paragraphs
    paragraphs = response_text.split('\n\n')
    formatted_paragraphs = []

    for paragraph in paragraphs:
        # Format bullet points
        if paragraph.startswith('- ') or paragraph.startswith('* '):
            lines = paragraph.split('\n')
            formatted_lines = [f"• {line[2:]}" for line in lines]
            formatted_paragraph = '\n'.join(formatted_lines)
        else:
            # Format bold text
            formatted_paragraph = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', paragraph)
        
        formatted_paragraphs.append(formatted_paragraph)

    # Join paragraphs with proper spacing
    formatted_response = '<br><br>'.join(formatted_paragraphs)

    # Add any additional formatting as needed
    formatted_response = formatted_response.replace('**', '<strong>').replace('</strong>**', '</strong>')

    return formatted_response

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002, debug=True)
