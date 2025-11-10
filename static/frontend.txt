"""
Semantic Similarity Web Application - Backend
Flask server that uses OpenAI embeddings for semantic similarity

Required Environment Variables:
- OPENAI_API_KEY: Your OpenAI API key

Install dependencies:
pip install flask flask-cors openai python-dotenv
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import openai
import os
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

if not openai.api_key:
    print("WARNING: OPENAI_API_KEY not found in environment variables!")
    print("Please create a .env file with: OPENAI_API_KEY=your-key-here")

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "api_key_set": bool(openai.api_key)})

@app.route('/calculate-similarity', methods=['POST'])
def calculate_similarity():
    """
    Calculate semantic similarity between documents
    
    Expected JSON:
    {
        "documents": [
            {"name": "Doc 1", "text": "..."},
            {"name": "Doc 2", "text": "..."}
        ]
    }
    
    Returns:
    {
        "success": true,
        "results": [
            {
                "doc1": "Doc 1",
                "doc2": "Doc 2", 
                "similarity": 0.85
            }
        ],
        "matrix": [[1.0, 0.85], [0.85, 1.0]]
    }
    """
    try:
        data = request.json
        documents = data.get('documents', [])
        
        # Validate input
        if len(documents) < 2:
            return jsonify({
                "success": False,
                "error": "Please provide at least 2 documents"
            }), 400
        
        # Extract texts and names
        texts = [doc['text'] for doc in documents]
        names = [doc['name'] for doc in documents]
        
        # Validate texts are not empty
        if any(not text.strip() for text in texts):
            return jsonify({
                "success": False,
                "error": "All documents must have text content"
            }), 400
        
        # Get embeddings from OpenAI
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-small",  # Cost-effective and accurate
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"OpenAI API error: {str(e)}"
            }), 500
        
        # Calculate cosine similarity matrix
        embeddings_array = np.array(embeddings)
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        normalized_embeddings = embeddings_array / norms
        
        # Compute similarity matrix
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        # Prepare results
        results = []
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity_score = float(similarity_matrix[i][j])
                results.append({
                    "doc1": names[i],
                    "doc2": names[j],
                    "similarity": similarity_score
                })
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return jsonify({
            "success": True,
            "results": results,
            "matrix": similarity_matrix.tolist(),
            "names": names
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)