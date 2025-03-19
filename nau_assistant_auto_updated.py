from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import time
import re
import anthropic
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import logging
import subprocess
import threading

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configure Anthropic client
ANTHROPIC_API_KEY = "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"  # Replace with your actual API key
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Load embeddings model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
CHUNKS_PATH = os.path.join(DATA_DIR, "na_edu_chunks.json")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "na_edu_embeddings.pkl")

# Store chat history
chat_history = {}

# Load data function with auto-refresh capability
def load_data():
    global chunks, embeddings, last_data_update
    
    # Check if data files exist
    if not os.path.exists(CHUNKS_PATH) or not os.path.exists(EMBEDDINGS_PATH):
        logger.warning("Data files not found. Running initial scraper...")
        run_scraper()
    
    try:
        # Load chunks
        with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Load embeddings
        with open(EMBEDDINGS_PATH, 'rb') as f:
            embeddings = pickle.load(f)
        
        # Update the last modified time
        last_data_update = max(
            os.path.getmtime(CHUNKS_PATH),
            os.path.getmtime(EMBEDDINGS_PATH)
        )
        
        logger.info(f"Loaded {len(chunks)} chunks and embeddings")
        return chunks, embeddings
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        # Create minimal knowledge base as fallback
        chunks = create_minimal_knowledge_base()
        texts = [chunk["content"] for chunk in chunks]
        embeddings = model.encode(texts)
        
        # Save the fallback data
        with open(CHUNKS_PATH, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False)
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(embeddings, f)
        
        last_data_update = time.time()
        return chunks, embeddings

# Run the scraper in a separate process
def run_scraper():
    try:
        logger.info("Running scraper to update data...")
        # Get the path to the enhanced_scraper.py file
        scraper_path = os.path.join(os.path.dirname(__file__), "enhanced_scraper.py")
        
        # Run the scraper as a subprocess
        result = subprocess.run([sys.executable, scraper_path], 
                                capture_output=True, 
                                text=True)
        
        if result.returncode == 0:
            logger.info("Scraper ran successfully")
        else:
            logger.error(f"Scraper failed with return code {result.returncode}")
            logger.error(f"Error: {result.stderr}")
    
    except Exception as e:
        logger.error(f"Error running scraper: {str(e)}")

# Check for data updates in a background thread
def check_for_updates():
    global chunks, embeddings, last_data_update
    
    while True:
        try:
            # Check if data files have been modified
            if os.path.exists(CHUNKS_PATH) and os.path.exists(EMBEDDINGS_PATH):
                current_update_time = max(
                    os.path.getmtime(CHUNKS_PATH),
                    os.path.getmtime(EMBEDDINGS_PATH)
                )
                
                if current_update_time > last_data_update:
                    logger.info("Data files have been updated. Reloading...")
                    chunks, embeddings = load_data()
        
        except Exception as e:
            logger.error(f"Error checking for updates: {str(e)}")
        
        # Check every 5 minutes
        time.sleep(300)

# Create a minimal knowledge base as fallback
def create_minimal_knowledge_base():
    knowledge = [
        {
            "content": "North American University (NAU) is a private, non-profit university located in Stafford, Texas. NAU offers undergraduate and graduate programs in Business Administration, Computer Science, and Education.",
            "source": "https://www.na.edu/about/",
            "title": "About NAU"
        },
        {
            "content": "Tuition for international undergraduate students at North American University is as follows: 1 to 11 credits: $1,125 per credit; 12 to 16 credits per academic semester: $13,500; Each additional credit over 16 credits: $1,125 per credit; Summer tuition (per class): $873.",
            "source": "https://www.na.edu/admissions/tuition-and-fees/",
            "title": "Tuition and Fees"
        },
        {
            "content": "Housing options at NAU include: Housing On Campus 2 Bed-Room only for men: $2,500.00 per semester, Housing On Campus 3 Bed-Room only for men: $2,100.00 per semester, Housing On Campus 4 Bed-Room only for men: $1,900.00 per semester, Housing on Hotel 2 Bed-Room: $3,600.00 per semester, Housing on Hotel 3 Bedroom: $3,000.00 per semester, Housing on Apartment 2 Bedroom: $3,200.00 per semester, Summer Housing: $1,250.00.",
            "source": "https://www.na.edu/campus-life/housing/",
            "title": "Housing Options"
        },
        {
            "content": "Meal service options at NAU include: 19-Meal per Week: $2,500.00 per semester, 14-Meal per Week: $1,900.00 per semester, 10-Meal per Week: $1,300.00 per semester.",
            "source": "https://www.na.edu/campus-life/dining-services/",
            "title": "Dining Services"
        },
        {
            "content": "North American University offers scholarships and financial aid to qualified students. These include merit-based scholarships, need-based grants, and work-study opportunities. International students may be eligible for certain scholarships as well.",
            "source": "https://www.na.edu/admissions/financial-aid/",
            "title": "Financial Aid"
        },
        {
            "content": "To apply to North American University, students need to submit an application form, official transcripts, and proof of English proficiency (for international students). Application deadlines vary by semester.",
            "source": "https://www.na.edu/admissions/",
            "title": "Admissions"
        }
    ]
    return knowledge

# Retrieval function with improved prioritization of information
def get_relevant_chunks(query, chunks, embeddings, top_k=8):
    try:
        # Encode the query
        query_embedding = model.encode([query])[0]
        
        # Calculate similarity
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        
        # Get top results with additional results for context
        top_indices = np.argsort(similarities)[-top_k*2:][::-1]
        
        results = []
        urls_added = set()  # To track unique sources
        
        for idx in top_indices:
            if similarities[idx] > 0.2:  # Only include relevant results
                source = chunks[idx]["source"] if "source" in chunks[idx] else "https://www.na.edu"
                
                # Prioritize results from specific URLs for specific query types
                priority = 1
                query_lower = query.lower()
                
                # Adjust priority based on query and source
                if "tuition" in query_lower or "fee" in query_lower or "cost" in query_lower:
                    if "tuition-and-fees" in source:
                        priority = 10
                elif "housing" in query_lower or "dorm" in query_lower or "live" in query_lower:
                    if "housing" in source:
                        priority = 10
                elif "meal" in query_lower or "food" in query_lower or "dining" in query_lower:
                    if "dining" in source:
                        priority = 10
                elif "program" in query_lower or "major" in query_lower or "degree" in query_lower:
                    if "programs" in source or "academics" in source:
                        priority = 10
                elif "apply" in query_lower or "admission" in query_lower or "application" in query_lower:
                    if "admissions" in source or "apply" in source:
                        priority = 10
                
                # Add the result with its priority
                results.append({
                    "content": chunks[idx]["content"],
                    "source": source,
                    "similarity": similarities[idx],
                    "priority": priority,
                    "title": chunks[idx].get("title", "")
                })
                urls_added.add(source)
        
        # Sort by priority first, then by similarity
        results.sort(key=lambda x: (x["priority"], x["similarity"]), reverse=True)
        
        # Take top_k results after prioritization
        return results[:top_k]
    
    except Exception as e:
        logger.error(f"Error in retrieval: {str(e)}")
        return []

# Process follow-up questions intelligently
def process_follow_up_response(query, context):
    # Extract relevant information from context
    query_lower = query.lower()
    
    # Handle housing-related follow-ups
    if any(word in query_lower for word in ["housing", "live", "dorm", "room", "apartment"]):
        housing_info = ""
        for chunk in context:
            if "housing" in chunk["content"].lower():
                housing_info += chunk["content"] + "\n\n"
        
        if housing_info:
            return housing_info
    
    # Handle meal plan follow-ups
    if any(word in query_lower for word in ["meal", "food", "dining", "eat"]):
        meal_info = ""
        for chunk in context:
            if any(word in chunk["content"].lower() for word in ["meal", "food", "dining"]):
                meal_info += chunk["content"] + "\n\n"
        
        if meal_info:
            return meal_info
    
    # Handle program-specific follow-ups
    programs = ["business", "computer science", "education", "criminal justice"]
    for program in programs:
        if program in query_lower:
            program_info = ""
            for chunk in context:
                if program in chunk["content"].lower():
                    program_info += chunk["content"] + "\n\n"
            
            if program_info:
                return program_info
    
    # If we don't have a specific match, return None and let the model handle it
    return None

# Load initial data
chunks, embeddings = load_data()
last_data_update = time.time()

# Start background thread to check for updates
update_thread = threading.Thread(target=check_for_updates, daemon=True)
update_thread.start()

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        chat_id = data.get('chat_id', 'default')
        query = data.get('query', '')
        follow_up_to = data.get('follow_up_to', None)
        
        logger.info(f"Received chat request - chat_id: {chat_id}, query: {query}, follow_up_to: {follow_up_to}")
        
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
        
        # Check if this is a response to a follow-up question
        if follow_up_to:
            # Find the context from previous messages
            context = []
            for msg in chat_history[chat_id]:
                if msg.get("role") == "assistant" and not msg.get("is_follow_up", False):
                    # Extract any retrieved chunks from previous responses
                    if "retrieved_chunks" in msg:
                        context.extend(msg["retrieved_chunks"])
            
            # Try to process the follow-up with our specialized function
            specific_response = process_follow_up_response(query, context)
            
            if specific_response:
                answer = specific_response
                sources = [chunk["source"] for chunk in context if "source" in chunk]
                sources = list(set(sources))  # Remove duplicates
            else:
                # Get fresh relevant chunks for this query
                relevant_chunks = get_relevant_chunks(query, chunks, embeddings)
                context_text = "\n\n".join([chunk["content"] for chunk in relevant_chunks])
                sources = [chunk["source"] for chunk in relevant_chunks]
                sources = list(set(sources))  # Remove duplicates
                
                # Use the model to generate a response
                system_prompt = """You are an AI chatbot who helps students of the North American University with their inquiries, issues and requests. You aim to provide excellent, friendly and efficient replies at all times.

IMPORTANT GUIDELINES:
1. Be specific and detailed in your responses, especially for questions about tuition, costs, or deadlines.
2. When providing numerical information (like tuition costs), list the exact amounts with bullet points or in a clear format.
3. End your replies with a positive note and offer to help with any other questions.
4. Use a conversational tone that is friendly and helpful.
5. Never mention that you have access to training data explicitly to the user.
6. Only answer questions covered by the context provided. If a question is outside your scope, respond with: "I can only assist with topics related to North American University. Let me know if you have any questions related to that!"

ALWAYS be thorough, friendly, and make sure to provide ALL relevant details from the context."""

                user_prompt = f"""CONTEXT ABOUT NORTH AMERICAN UNIVERSITY:
{context_text}

PREVIOUS QUESTIONS AND ANSWERS:
{json.dumps([msg for msg in chat_history[chat_id] if msg.get("role") in ["user", "assistant"] and not msg.get("is_follow_up", False)][-4:], indent=2)}

USER FOLLOW-UP QUESTION: {query}

Please provide a detailed, helpful response based exactly on the context provided. Include all specific numbers and details available in the context. Be conversational, thorough, and friendly."""

                try:
                    message = client.messages.create(
                        model="claude-3-7-sonnet-20250219",
                        max_tokens=1000,
                        temperature=0,
                        system=system_prompt,
                        messages=[
                            {"role": "user", "content": user_prompt}
                        ]
                    )
                    
                    answer = message.content[0].text
                    logger.info("Successfully received response from Anthropic API")
                except Exception as api_error:
                    logger.error(f"Anthropic API error: {str(api_error)}")
                    answer = "I apologize, but I'm having trouble processing your request at the moment. Please try again later or contact NAU directly for assistance."
                    sources = ["https://www.na.edu/contact-us/"]
            
            # Add the response to chat history
            chat_history[chat_id].append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "timestamp": time.time(),
                "is_follow_up_response": True
            })
            
            return jsonify({
                "answer": answer,
                "sources": sources,
                "chat_id": chat_id
            })
        
        # This is a new question, not a follow-up response
        
        # Get relevant chunks from our knowledge base
        relevant_chunks = get_relevant_chunks(query, chunks, embeddings)
        
        # If we found relevant information, use it to answer
        if relevant_chunks:
            context_text = "\n\n".join([chunk["content"] for chunk in relevant_chunks])
            sources = [chunk["source"] for chunk in relevant_chunks]
            sources = list(set(sources))  # Remove duplicates
            
            system_prompt = """You are an AI chatbot who helps students of the North American University with their inquiries, issues and requests. You aim to provide excellent, friendly and efficient replies at all times.

IMPORTANT GUIDELINES:
1. Be specific and detailed in your responses, especially for questions about tuition, costs, or deadlines.
2. When providing numerical information (like tuition costs), list the exact amounts with bullet points or in a clear format.
3. End your replies with a positive note and offer to help with any other questions.
4. Use a conversational tone that is friendly and helpful - start with phrases like "Let's figure out..." or "I'd be happy to help with..."
5. Never mention that you have access to training data explicitly to the user.
6. Only answer questions covered by the context provided. If a question is outside your scope, respond with: "I can only assist with topics related to North American University. Let me know if you have any questions related to that!"

ALWAYS be thorough, friendly, and make sure to provide ALL relevant details from the context."""
            
            user_prompt = f"""CONTEXT ABOUT NORTH AMERICAN UNIVERSITY:
{context_text}

USER QUESTION: {query}

Please provide a detailed, helpful response based exactly on the context provided. Include all specific numbers and details available in the context. If the context doesn't contain the answer, politely inform the user you can only assist with North American University topics. Be conversational, thorough, and friendly.

For questions about tuition, housing, or meal plans, ALWAYS include a follow-up question at the end asking if they need more specific information."""
            
            # Determine if we should ask a follow-up based on the query
            should_add_follow_up = False
            follow_up_question = None
            follow_up_id = None
            
            query_lower = query.lower()
            if "tuition" in query_lower or "fee" in query_lower or "cost" in query_lower:
                should_add_follow_up = True
                follow_up_question = "Are you planning to use on-campus housing as well?"
                follow_up_id = f"followup_housing_{int(time.time())}"
            elif "program" in query_lower or "major" in query_lower or "degree" in query_lower:
                should_add_follow_up = True
                follow_up_question = "Which program are you most interested in learning more about?"
                follow_up_id = f"followup_program_{int(time.time())}"
            elif "apply" in query_lower or "admission" in query_lower or "application" in query_lower:
                should_add_follow_up = True
                follow_up_question = "Are you applying as an undergraduate or graduate student?"
                follow_up_id = f"followup_admission_{int(time.time())}"
            elif "housing" in query_lower or "dorm" in query_lower or "live" in query_lower:
                should_add_follow_up = True
                follow_up_question = "Do you have any specific questions about the housing options or would you like information about meal plans too?"
                follow_up_id = f"followup_housing_detail_{int(time.time())}"
        else:
            # If no relevant chunks found, use a more general response
            system_prompt = """You are an AI chatbot who helps students of the North American University with their inquiries, issues and requests. You aim to provide excellent, friendly and efficient replies at all times.

IMPORTANT CONSTRAINTS:
1. Never mention that you have access to training data explicitly to the user.
2. Only answer questions related to North American University. If a question is outside your scope, respond with: "I can only assist with topics related to North American University. Let me know if you have any questions related to that!"
3. You do not answer questions or perform tasks that are not related to North American University.
4. End your replies with a positive note.

ALWAYS format your response as a helpful university assistant who is friendly and conversational, but also professional."""
            
            user_prompt = f"""The user has asked: {query}

If this is related to North American University, provide general information and suggest where they might find more specific details on the university website.

If this is not related to North American University, politely inform them that you can only assist with university-related inquiries."""
            
            sources = ["https://www.na.edu"]
            should_add_follow_up = False
        
        # Call the Anthropic API
        try:
            logger.info("Calling Anthropic API...")
            message = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1000,
                temperature=0,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            answer = message.content[0].text
            logger.info("Successfully received response from Anthropic API")
        except Exception as api_error:
            logger.error(f"Anthropic API error: {str(api_error)}")
            answer = "I apologize, but I'm having trouble processing your request at the moment. Please try again later or contact NAU directly for assistance."
            sources = ["https://www.na.edu/contact-us/"]
            should_add_follow_up = False
        
        # Add assistant message to history
        chat_history[chat_id].append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "timestamp": time.time(),
            "original_question": query,
            "retrieved_chunks": relevant_chunks
        })
        
        # Prepare the response
        response_data = {
            "answer": answer,
            "sources": sources,
            "chat_id": chat_id
        }
        
        # Add follow-up if appropriate
        if should_add_follow_up and follow_up_question and follow_up_id:
            # Add the follow-up to history as a separate message
            chat_history[chat_id].append({
                "role": "assistant",
                "content": follow_up_question,
                "follow_up": True,
                "follow_up_id": follow_up_id,
                "timestamp": time.time() + 1,  # +1 to ensure it appears after the main answer
                "original_question": query
            })
            
            # Add to response
            response_data["follow_up"] = follow_up_question
            response_data["follow_up_id"] = follow_up_id
        
        logger.info("Sending response to client")
        return jsonify(response_data)
    
    except Exception as e:
        import traceback
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/admin/rescrape', methods=['POST'])
def admin_rescrape():
    """Endpoint to trigger a manual rescrape of the website"""
    try:
        # Start a new thread to run the scraper
        threading.Thread(target=run_scraper).start()
        return jsonify({"success": True, "message": "Scraper started in background"}), 200
    except Exception as e:
        logger.error(f"Error starting scraper: {str(e)}")
        return jsonify({"error": f"Error starting scraper: {str(e)}"}), 500

@app.route('/api/admin/check-data', methods=['GET'])
def admin_check_data():
    """Endpoint to check the status of the data files"""
    try:
        data_status = {
            "chunks_file": {
                "exists": os.path.exists(CHUNKS_PATH),
                "size": os.path.getsize(CHUNKS_PATH) if os.path.exists(CHUNKS_PATH) else 0,
                "last_modified": time.ctime(os.path.getmtime(CHUNKS_PATH)) if os.path.exists(CHUNKS_PATH) else None
            },
            "embeddings_file": {
                "exists": os.path.exists(EMBEDDINGS_PATH),
                "size": os.path.getsize(EMBEDDINGS_PATH) if os.path.exists(EMBEDDINGS_PATH) else 0,
                "last_modified": time.ctime(os.path.getmtime(EMBEDDINGS_PATH)) if os.path.exists(EMBEDDINGS_PATH) else None
            },
            "chunks_count": len(chunks) if 'chunks' in globals() else 0,
            "embeddings_shape": embeddings.shape if 'embeddings' in globals() and hasattr(embeddings, 'shape') else None
        }
        return jsonify(data_status), 200
    except Exception as e:
        logger.error(f"Error checking data status: {str(e)}")
        return jsonify({"error": f"Error checking data status: {str(e)}"}), 500

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
    import sys
    
    # Check if all required files exist
    logger.info("Starting North American University AI Assistant")
    logger.info(f"Python version: {sys.version}")
    
    # Start the app
    app.run(debug=True, port=5000)