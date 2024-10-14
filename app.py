from flask import Flask, render_template, request, jsonify
import os
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from flask_cors import CORS

# Configure Google Generative AI
GENAI_API_KEY = "AIzaSyDceI3mqdAoSIPkGpYgbttbuJ-YUwoLk3E"
genai.configure(api_key=GENAI_API_KEY)

# Configure Flask app
logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

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

@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        user_query = request.form.get("query")
        if user_query and tfidf_matrix is not None:
            # Transform the user query into TF-IDF vector
            query_vec = vectorizer.transform([user_query])

            # Compute cosine similarity between query and all documents
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

            # Get indices of top N similar documents
            top_indices = similarities.argsort()[-TOP_N:][::-1]

            # Retrieve the content of top N documents
            retrieved_docs = [documents[idx] for idx in top_indices]
            retrieved_content = "\n\n".join(retrieved_docs)

            # Prepare the prompt for Gemini API with retrieved context
            prompt = f"Context:\n{retrieved_content}\n\nQuestion: {user_query}\nAnswer:"

            try:
                # Generate response using the Generative AI library
                api_response = model.generate_content(prompt)
                response = format_response(api_response.text.strip())
            except Exception as e:
                response = f"An error occurred while generating the response: {e}"
        elif tfidf_matrix is None:
            response = "The knowledge base is empty. Please add .txt files to the 'knowledge_base' directory."
        else:
            response = "Please enter a valid query."

    return render_template("index.html", response=response)

def format_response(response_text):
    # Improve the formatting of the response
    formatted_response = response_text.replace("* **", "\n- **")
    return formatted_response

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)