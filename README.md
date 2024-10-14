Text Processing with TF-IDF: The knowledge base is converted into numerical values using TF-IDF, which helps transform text into a format that a machine can easily work with.

User Query Conversion & Matching: Your question is also converted using TF-IDF, allowing the system to efficiently compare your query with the knowledge base using Cosine Similarity to find the best match.

Generating a Detailed Response: The matched content is sent to the Gemini API to generate a comprehensive response, enhancing the quality and relevance of the answer.

Add Gemini API Key in app.py and you are good to go.
