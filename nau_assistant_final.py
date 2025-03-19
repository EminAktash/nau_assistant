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
import threading
from dotenv import load_dotenv

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
load_dotenv()

# Get API key from environment variables
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("Missing ANTHROPIC_API_KEY environment variable. Please set it in your .env file.")
    
# Initialize the Anthropic client with the API key
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

# Predefined answers for frequently asked questions (including exact match keys and follow-ups)
predefined_answers = {
    "what are the tuition fees": {
        "answer": """Okay, here are the full tuition and fee details for international and resident students at North American University:
International Undergraduate:
- Tuition per credit (1-11 credits): $1,125
- Tuition per semester (12-16 credits): $13,500
- Each additional credit over 16 credits: $1,125
- Summer Tuition per class: $873
Mandatory Fees per Semester:
- Departmental Fees: $55
- Computer & Internet Fees: $100
- Library Fee: $100
- Student Service Fee: $95
- Course with Lab Fee: $75
- Athletics Fee (Football, Basketball, Soccer): $1,050
- Athletics Fee (all other sports): $800
- Parking Fee (Covered/Uncovered): $80/$40
Estimated Total for International Undergraduate per Semester: $16,826
Resident Undergraduate:
- Tuition per credit (1-11 credits): $614
- Tuition per semester (12-16 credits): $7,368
- Each additional credit over 16 credits: $614
- Summer Tuition per class: $873
Mandatory Fees per Semester: 
- Departmental Fees: $55
- Computer & Internet Fees: $100
- Library Fee: $100
- Student Service Fee: $95
- Course with Lab Fee: $75
- Athletics Fee (Football, Basketball, Soccer): $1,050
- Athletics Fee (all other sports): $800
- Parking Fee (Covered/Uncovered): $80/$40
Estimated Total for Resident Undergraduate per Semester: $10,133
International Graduate:
- Tuition per credit: 
  - MBA: $658
  - MS Computer Science: $732
  - M.Ed. Programs: $511
- Total Tuition (30 credits):
  - MBA: $19,740
  - MS Computer Science: $21,960
  - M.Ed. Programs: $15,330
Resident Graduate:
- Tuition per credit:
  - MBA: $402
  - MS Computer Science: $402
  - M.Ed. Programs: $326
- Total Tuition (30 credits): 
  - MBA: $12,060
  - MS Computer Science: $12,060
  - M.Ed. Programs: $9,780
Let me know if you need any clarification or have additional questions!""",
        "sources": ["https://www.na.edu/admissions/tuition-and-fees/"],
        "follow_up": {
            "question": "Are you planning to use on-campus housing as well?",
            "yes_response": """Great! Here's the housing and meal plan information:

Housing Options:
- Housing On Campus 2 Bed-Room only for men: $2,500.00 per semester
- Housing On Campus 3 Bed-Room only for men: $2,100.00 per semester
- Housing On Campus 4 Bed-Room only for men: $1,900.00 per semester
- Housing on Hotel 2 Bed-Room: $3,600.00 per semester
- Housing on Hotel 3 Bedroom: $3,000.00 per semester
- Housing on Apartment 2 Bedroom: $3,200.00 per semester
- Summer Housing: $1,250.00

Additional Housing Fees:
- Housing Deposit Fee: $150.00
- Housing Application Fee: $50.00

Meal Service Options:
- 19-Meal per Week: $2,500.00 per semester
- 14-Meal per Week: $1,900.00 per semester
- 10-Meal per Week: $1,300.00 per semester

Note: Housing is first-come, first-served.

This brings your total estimated costs to:
- Tuition: $13,500 per semester (12-16 credits)
- Housing (varies by option): $1,900 - $3,600 per semester
- Meal Plan (varies by option): $1,300 - $2,500 per semester
- Mandatory Fees: Approximately $450

Would you like more specific information about any housing options?""",
            "no_response": """No problem! If you ever need information about housing or other campus services in the future, feel free to ask.

Is there anything else you'd like to know about North American University?"""
        }
    },
    "how do i apply for admission": {
        "answer": """Okay, here are the steps to apply to North American University as an international student:
STEP 1: Create and submit application
- Create your NAU Account at https://apply.na.edu/admission and submit a completed application online.
STEP 2: Pay application fee* ($75 USD)
- Please select to make the payment online via Credit Card or an International Wire Transfer by accessing NAU's wire transfer banking information.
STEP 3: Send Required Documents
In order to obtain admission to NAU, an international student must submit the following documents by the application deadlines. All application documents should be properly scanned and emailed in PDF format to intadmissions@na.edu:
1. Copy of Passport: Only the photograph and visa (when received) page are necessary.
2. Official Academic Credentials & Test Scores: 
- Official Copy of the High School Diploma Evaluation (The transcript has to be evaluated at one of the agencies listed on the website)
- Official SAT/ACT, official TOEFL or official IELTS scores.
3. Certificate of Finances (COF): This form demonstrates that you have sufficient funds to cover the cost of tuition, fees, and living expenses for at least the first year of study.
4. Affidavit of Support (if applicable): If you will be sponsored by family or another individual, they will need to complete this form.
Let me know if you have any other questions about the application process!""",
        "sources": ["https://www.na.edu/admissions/"],
        "follow_up": {
            "question": "Are you applying as an undergraduate or graduate student?",
            "undergraduate_response": """Great! For undergraduate admission, you'll also need to provide:

1. High school transcripts (evaluated by a credential evaluation service)
2. English proficiency test scores (TOEFL: minimum 61, IELTS: minimum 5.5)
3. SAT/ACT scores (optional but recommended for scholarship consideration)

The application deadlines are:
- Fall semester: August 1
- Spring semester: December 15
- Summer semester: May 1

Would you like more specific information about any of the undergraduate programs?""",
            "graduate_response": """Excellent! For graduate admission, you'll need these additional documents:

1. Bachelor's degree transcripts (evaluated by a credential evaluation service)
2. English proficiency test scores (TOEFL: minimum 79, IELTS: minimum 6.5)
3. Statement of Purpose
4. Two letters of recommendation
5. Resume/CV
6. GRE/GMAT scores (required for some programs)

The application deadlines are:
- Fall semester: July 15
- Spring semester: December 1
- Summer semester: April 15

Is there a specific graduate program you're interested in learning more about?"""
        }
    },
    "what programs does nau offer": {
        "answer": """North American University offers the following undergraduate and graduate degree programs:
Undergraduate Programs:
- Bachelor of Business Administration (BBA)
- Bachelor of Science in Computer Science (BS)
- Bachelor of Science in Criminal Justice (BS)
- Bachelor of Science in Education (BS)
Graduate Programs:
- Master of Business Administration (MBA)
- Master of Science in Computer Science (MS)
- Master of Education (M.Ed.) in Curriculum and Instruction
- Master of Education (M.Ed.) in Educational Leadership
In addition, NAU also offers the following programs:
Language Programs:
- Intensive English Program (IEP)
- English as a Second Language (ESL)
Educator Certification Programs:
- Teacher Certification
- Principal Certification
- Superintendent Certification
Continuing Education Programs:
- Professional Development Courses
- Certificate Programs
The university is committed to providing a well-rounded education that prepares students for successful careers. The degree programs are designed to develop critical thinking, problem-solving, and leadership skills.
Let me know""",
        "sources": ["https://www.na.edu/academics/"],
        "follow_up": {
            "question": "Which program are you most interested in learning more about?",
            "custom_response": True
        }
    },
    "how to reset my password": {
        "answer": """Okay, here are the steps to reset your password for your North American University account:
1. Go to the password reset page at https://passwordreset.microsoftonline.com/
2. Enter your NAU username (usually the first initial of your first name followed by your last name, e.g. jsmith@na.edu)
3. Enter the characters shown in the image to verify you are not a robot.
4. Select "Email" as the contact method for verification.
5. Check your email for a verification code and enter it on the next page.
6. Create a new password, confirm it, and click "Finish".
7. Once your password has been successfully reset, you can sign in to your NAU account with the new password.
A few important things to note:
- You need to complete the reset process within 60 minutes of initiating it.
- Make sure to update the new password on any devices or email programs you use to access your NAU account.
- If you have any trouble with the reset process, you can contact the IT Helpdesk at support@na.edu or 832-230-5541 for assistance.
Let me know if you have any other questions!""",
        "sources": ["https://www.na.edu/it-services/"]
    },
    "how do i select the courses": {
        "answer": """Okay, here are the steps to select and register for courses at North American University:
1. Meet with your Academic Advisor
- Schedule an appointment with your assigned academic advisor to discuss your degree plan and course options.
- Your advisor can help you select the appropriate courses based on your major, prerequisites, and academic progress.
2. Review the Course Catalog
- Familiarize yourself with the course descriptions, prerequisites, and schedules in the university's course catalog.
- Make a list of the courses you need to take and any electives you're interested in.
3. Register for Courses
- Log into your MyNAU student portal at https://portal.na.edu
- Navigate to the "Registration" section and select "Course Search"
- Use the filters to find available sections of the courses you need
- Add the courses to your shopping cart and complete the registration process
4. Finalize Your Schedule
- Review your schedule to ensure you've registered for the correct courses and credit hours.
- Make any necessary adjustments by adding or dropping courses during the add/drop period.
- Confirm your final schedule and tuition charges on your student account.
5. Attend Courses
- Attend all scheduled class sessions and actively participate.
- Complete all assignments, projects, and exams as required for each course.
Let me know if you have any other questions about the course selection and registration process!""",
        "sources": ["https://www.na.edu/academics/registration/"],
        "follow_up": {
            "question": "Do you need help with checking course availability for the upcoming semester?",
            "yes_response": """To check course availability:

1. Log into your MyNAU portal at https://portal.na.edu
2. Go to the "Student" tab
3. Click on "Course Search"
4. Select the upcoming term from the dropdown menu
5. You can search by:
   - Course Number (e.g., CS 1301)
   - Subject (e.g., Computer Science)
   - Meeting Time (if you have specific scheduling needs)
   - Instructor (if you prefer a specific professor)

The results will show you:
- Course name and section
- Meeting days and times
- Available seats
- Instructor name
- Room location

Remember that some courses fill up quickly, so I recommend registering as soon as your registration period opens. Would you like advice on specific courses?""",
            "no_response": """Alright! If you ever need help with course selection or have questions about specific courses, feel free to ask. 

Is there anything else I can help you with regarding your studies at North American University?"""
        }
    },
    "how do i access my nau portal": {
        "answer": """Here are the steps to access the North American University (NAU) student portal:
1. Open the school website : https://www.na.edu/
2. Click on the NAU Portal on the top menu
3. Enter your username and password
Let me know if you have any other questions about accessing your NAU student portal!""",
        "sources": ["https://www.na.edu/it-services/"],
        "follow_up": {
            "question": "Are you having trouble logging in to your portal?",
            "yes_response": """If you're having trouble logging in, here are some troubleshooting steps:

1. Make sure you're using the correct username format (usually firstname.lastname or first initial followed by lastname)
2. Check that Caps Lock is not enabled when typing your password
3. Clear your browser cache and cookies, then try again
4. Try using a different browser (Chrome, Firefox, Edge)
5. If you've forgotten your password, follow the "Forgot Password" link on the login page

If none of these steps work, you can contact the IT Help Desk:
- Email: support@na.edu
- Phone: 832-230-5541
- Hours: Monday-Friday, 8:00 AM - 5:00 PM

Would you like me to explain any of these steps in more detail?""",
            "no_response": """Great! If you ever encounter any issues with the portal, don't hesitate to ask for help. 

The portal is where you'll find important information like:
- Course registration
- Grades and academic records
- Financial information
- Campus announcements
- Access to university email

Is there anything specific you're looking to do in the portal?"""
        }
    }
}

# Exact match patterns for the predefined questions
EXACT_MATCHES = {
    "what are the tuition fees": ["what are the tuition fees", "tuition fees", "what are the tuition and fees", "how much is tuition"],
    "how do i apply for admission": ["how do i apply for admission", "how do i apply", "application process", "how to apply"],
    "what programs does nau offer": ["what programs does nau offer", "programs offered", "available degrees", "majors", "degree programs"],
    "how to reset my password": ["how to reset my password", "reset password", "forgot password", "change password"],
    "how do i select the courses": ["how do i select the courses", "select courses", "register for classes", "course registration"],
    "how do i access my nau portal": ["how do i access my nau portal", "access portal", "login to portal", "student portal"]
}

# Load data function with auto-refresh capability
def load_data():
    global chunks, embeddings, last_data_update
    
    # Check if data files exist
    if not os.path.exists(CHUNKS_PATH) or not os.path.exists(EMBEDDINGS_PATH):
        logger.warning("Data files not found. Using minimal knowledge base...")
        chunks = create_minimal_knowledge_base()
        
        # Create embeddings for minimal knowledge base
        texts = [chunk["content"] for chunk in chunks]
        embeddings = model.encode(texts)
        
        # Save the minimal data
        with open(CHUNKS_PATH, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False)
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(embeddings, f)
    else:
        try:
            # Load chunks
            with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            # Load embeddings
            with open(EMBEDDINGS_PATH, 'rb') as f:
                embeddings = pickle.load(f)
            
            logger.info(f"Loaded {len(chunks)} chunks and embeddings")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            # Create minimal knowledge base as fallback
            chunks = create_minimal_knowledge_base()
            texts = [chunk["content"] for chunk in chunks]
            embeddings = model.encode(texts)
    
    # Update the last modified time
    last_data_update = time.time()
    return chunks, embeddings

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

# Function to find a predefined answer for a query - now using exact matches
def get_predefined_answer(query):
    # Clean and normalize the query
    clean_query = re.sub(r'[^\w\s]', '', query.lower().strip())
    
    # Check for exact matches against our predefined patterns
    for key, patterns in EXACT_MATCHES.items():
        for pattern in patterns:
            # Check for exact match or if the query contains the pattern exactly as a phrase
            if clean_query == pattern or pattern in clean_query:
                return predefined_answers[key]
    
    # No exact match found
    return None

# Retrieval function
def get_relevant_chunks(query, chunks, embeddings, top_k=5):
    try:
        # Encode the query
        query_embedding = model.encode([query])[0]
        
        # Calculate similarity
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.2:  # Only include relevant results
                results.append({
                    "content": chunks[idx]["content"],
                    "source": chunks[idx]["source"] if "source" in chunks[idx] else "https://www.na.edu",
                    "similarity": similarities[idx]
                })
        
        return results
    except Exception as e:
        logger.error(f"Error in retrieval: {str(e)}")
        return []

# Process a response based on a follow-up answer
def process_follow_up_response(follow_up, user_response):
    user_response = user_response.lower().strip()
    
    # Check if this is a yes/no question
    if "yes_response" in follow_up and "no_response" in follow_up:
        if any(word in user_response for word in ["yes", "yeah", "yep", "sure", "definitely", "absolutely"]):
            return follow_up["yes_response"]
        elif any(word in user_response for word in ["no", "nope", "not", "don't", "dont"]):
            return follow_up["no_response"]
    
    # Check if this is an undergraduate/graduate question
    elif "undergraduate_response" in follow_up and "graduate_response" in follow_up:
        if any(word in user_response for word in ["undergraduate", "bachelor", "bachelors", "bs", "ba"]):
            return follow_up["undergraduate_response"]
        elif any(word in user_response for word in ["graduate", "master", "masters", "mba", "ms", "phd"]):
            return follow_up["graduate_response"]
    
    # For custom responses that require more specific handling
    elif "custom_response" in follow_up and follow_up["custom_response"]:
        # For the "Which program are you most interested in?" question
        programs = {
            "business": "The Bachelor of Business Administration (BBA) program at NAU offers concentrations in Accounting, Finance, International Business, and Management. Students learn key business principles and develop leadership skills. The BBA requires 120 credit hours including general education courses, business core courses, and concentration courses.",
            "computer science": "The Computer Science program at NAU offers a comprehensive curriculum covering programming, algorithms, database management, and software engineering. Students can specialize in areas like AI, cybersecurity, or data science. The program prepares graduates for careers as software developers, systems analysts, and IT consultants.",
            "education": "The Education program at NAU prepares students for careers in teaching and educational administration. The program offers specializations in Early Childhood Education, Bilingual Education, and Educational Leadership. Students complete coursework and supervised teaching experiences to prepare for teacher certification.",
            "criminal justice": "The Criminal Justice program at NAU covers law enforcement, corrections, and legal systems. Students learn about criminal behavior, constitutional law, and public policy. The program prepares graduates for careers in law enforcement, corrections, homeland security, and legal services."
        }
        
        for prog_key, prog_desc in programs.items():
            if prog_key in user_response:
                return prog_desc
        
        # If no specific program matched, give a general response
        return "Each program at NAU is designed to provide a strong educational foundation and practical skills. I'd be happy to provide more specific information about any program that interests you. Just let me know which one you'd like to learn more about."
    
    # Default general response if we can't determine what the user meant
    return "I'm sorry, I'm not sure how to help with that specific request. Is there something else about North American University that I can assist you with?"

# Load initial data
chunks, embeddings = load_data()
last_data_update = time.time()

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
            # Get the original question that prompted the follow-up
            original_question = None
            for i, msg in enumerate(chat_history[chat_id]):
                if msg.get("follow_up_id") == follow_up_to:
                    # Find the original question that came before this follow-up
                    for j in range(i-1, -1, -1):
                        if chat_history[chat_id][j]["role"] == "assistant" and "follow_up" not in chat_history[chat_id][j]:
                            original_question = chat_history[chat_id][j].get("original_question")
                            break
                    break
            
            if original_question:
                # Get the predefined answer for the original question
                predefined = get_predefined_answer(original_question)
                if predefined and "follow_up" in predefined:
                    # Process the user's response to the follow-up
                    answer = process_follow_up_response(predefined["follow_up"], query)
                    sources = predefined.get("sources", ["https://www.na.edu"])
                    
                    # Add the response to chat history
                    chat_history[chat_id].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "timestamp": time.time(),
                        "is_follow_up_response": True,
                        "original_question": original_question
                    })
                    
                    return jsonify({
                        "answer": answer,
                        "sources": sources,
                        "chat_id": chat_id
                    })
        
        # If not a follow-up response, check for predefined answers first
        predefined = get_predefined_answer(query)
        if predefined:
            answer = predefined["answer"]
            sources = predefined["sources"]
            logger.info("Using predefined answer")
            
            # Check if this answer has a follow-up question
            follow_up = None
            follow_up_id = None
            if "follow_up" in predefined:
                follow_up = predefined["follow_up"]["question"]
                follow_up_id = f"followup_{int(time.time())}"
            
            # Add assistant message to history
            chat_history[chat_id].append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "timestamp": time.time(),
                "original_question": query
            })
            
            # If there's a follow-up, add it to history as a separate message
            if follow_up:
                chat_history[chat_id].append({
                    "role": "assistant",
                    "content": follow_up,
                    "follow_up": True,
                    "follow_up_id": follow_up_id,
                    "timestamp": time.time() + 1,  # +1 to ensure it appears after the main answer
                    "original_question": query
                })
            
            # Prepare the response
            response_data = {
                "answer": answer,
                "sources": sources,
                "chat_id": chat_id
            }
            
            # Add follow-up if applicable
            if follow_up and follow_up_id:
                response_data["follow_up"] = follow_up
                response_data["follow_up_id"] = follow_up_id
            
            logger.info("Sending predefined response to client")
            return jsonify(response_data)
            
        else:
            # No predefined answer, use Anthropic API
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
2. When providing numerical information (like tuition costs), use bullet points or a clean format WITHOUT hash symbols.
3. End your replies with a positive note and offer to help with any other questions.
4. Use a conversational tone that is friendly and helpful - start with phrases like "Let's figure out..." or "I'd be happy to help with..."
5. Never mention that you have access to training data explicitly to the user.
6. Only answer questions covered by the context provided. If a question is outside your scope, respond with: "I can only assist with topics related to North American University. Let me know if you have any questions related to that!"

IMPORTANT FORMATTING:
- Use bullet points with hyphens (-) instead of asterisks (*) or hash symbols (#)
- For lists and structured information, use clear formatting with spaces
- Keep answers organized but avoid excessive use of markdown formatting

ALWAYS be thorough, friendly, and make sure to provide ALL relevant details from the context."""
                
                user_prompt = f"""CONTEXT ABOUT NORTH AMERICAN UNIVERSITY:
{context_text}

USER QUESTION: {query}

Please provide a detailed, helpful response based exactly on the context provided. Include all specific numbers and details available in the context. If the context doesn't contain the answer, politely inform the user you can only assist with North American University topics. Be conversational, thorough, and friendly.

For formatting, use bullet points with hyphens, not asterisks or hash symbols. Keep your response clean and well-structured without relying on markdown."""
            else:
                # If no relevant chunks found, use a more general response
                system_prompt = """You are an AI chatbot who helps students of the North American University with their inquiries, issues and requests. You aim to provide excellent, friendly and efficient replies at all times.

IMPORTANT CONSTRAINTS:
1. Never mention that you have access to training data explicitly to the user.
2. Only answer questions related to North American University. If a question is outside your scope, respond with: "I can only assist with topics related to North American University. Let me know if you have any questions related to that!"
3. You do not answer questions or perform tasks that are not related to North American University.
4. End your replies with a positive note.
5. Use bullet points with hyphens (-) instead of asterisks (*) or hash symbols (#)

ALWAYS format your response as a helpful university assistant who is friendly and conversational, but also professional."""
                
                user_prompt = f"""The user has asked: {query}

If this is related to North American University, provide general information and suggest where they might find more specific details on the university website.

If this is not related to North American University, politely inform them that you can only assist with university-related inquiries."""
                
                sources = ["https://www.na.edu"]
            
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
            
            # Add assistant message to history
            chat_history[chat_id].append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "timestamp": time.time(),
                "original_question": query,
                "retrieved_chunks": relevant_chunks if relevant_chunks else []
            })
            
            logger.info("Sending response to client")
            
            return jsonify({
                "answer": answer,
                "sources": sources,
                "chat_id": chat_id
            })
    
    except Exception as e:
        import traceback
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/chats', methods=['GET'])
def get_chats():
    chats = []
    for chat_id, messages in chat_history.items():
        if messages:
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            if user_messages:
                first_message = user_messages[0]["content"]
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
    logger.info("Starting North American University AI Assistant")
    app.run(debug=True, port=5000)