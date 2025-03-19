from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import time
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

app = Flask(__name__)
CORS(app)

# Load your Anthropic API key
os.environ["ANTHROPIC_API_KEY"] = "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"  # Replace with your actual API key

# Initialize the embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load the vector store
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Initialize the Claude model
llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)

# Create a retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# Store chat history
chat_history = {}

# Serve the main index.html file
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# Serve other static files (like script.js)
@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    chat_id = data.get('chat_id', 'default')
    query = data.get('query', '')
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    # Initialize chat history if it doesn't exist
    if chat_id not in chat_history:
        chat_history[chat_id] = []
    
    # Add user message to history
    chat_history[chat_id].append({
        "role": "user",
        "content": query,
        "timestamp": time.time()
    })
    
    try:
        # Get response from QA chain
        result = qa_chain({"query": query})
        answer = result["result"]
        sources = [doc.metadata["source"] for doc in result["source_documents"]]
        
        # Remove duplicates from sources
        sources = list(set(sources))
        
        # Add system message to history
        chat_history[chat_id].append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "timestamp": time.time()
        })
        
        return jsonify({
            "answer": answer,
            "sources": sources,
            "chat_id": chat_id
        })
    
    except Exception as e:
        import traceback
        print(f"Error processing query: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/chats', methods=['GET'])
def get_chats():
    chats = []
    for chat_id, messages in chat_history.items():
        if messages:
            first_message = messages[0]["content"]
            preview = first_message[:50] + "..." if len(first_message) > 50 else first_message
            last_timestamp = messages[-1]["timestamp"]
            chats.append({
                "id": chat_id,
                "preview": preview,
                "timestamp": last_timestamp
            })
    
    # Sort by timestamp (newest first)
    chats.sort(key=lambda x: x["timestamp"], reverse=True)
    return jsonify(chats)

@app.route('/api/chats/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    if chat_id in chat_history:
        return jsonify(chat_history[chat_id])
    return jsonify([])

@app.route('/api/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    if chat_id in chat_history:
        del chat_history[chat_id]
    return jsonify({"success": True})

@app.route('/api/chats', methods=['POST'])
def create_chat():
    chat_id = f"chat_{int(time.time())}"
    chat_history[chat_id] = []
    return jsonify({"chat_id": chat_id})

if __name__ == '__main__':
    app.run(debug=True, port=5000)