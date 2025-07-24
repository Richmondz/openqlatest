# C:\Users\fishe\OneDrive\Documents\Project\OpenQProjects\Business Consulting\ai_routes.py
import os
import json
import numpy as np
import logging
import time
import random
from flask import Blueprint, request, jsonify, Response
from openai import OpenAI
from dotenv import load_dotenv
from flask_login import login_required, current_user
from models import Session, AICache

logger = logging.getLogger(__name__)
load_dotenv()

try:
    client = OpenAI()
    logger.info("OpenAI client initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {e}", exc_info=True)
    client = None

ai_bp = Blueprint('ai_bp', __name__)

# --- Your RAG and helper functions remain the same ---
knowledge_base_items = []
EMBEDDINGS_FILE_PATH = os.path.join('data', 'knowledge_base_with_embeddings.json')

def get_embedding(text_to_embed, model="text-embedding-3-small"):
    # (function is unchanged)
    if not client: return None
    try:
        text_to_embed = str(text_to_embed).replace("\n", " ")
        if not text_to_embed.strip(): return None
        response = client.embeddings.create(input=[text_to_embed], model=model)
        return response.data[0].embedding
    except Exception: return None

def load_precomputed_knowledge():
    # (function is unchanged)
    global knowledge_base_items
    if not os.path.exists(EMBEDDINGS_FILE_PATH):
        logger.error(f"Embeddings file not found: {EMBEDDINGS_FILE_PATH}")
        return
    try:
        with open(EMBEDDINGS_FILE_PATH, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        temp_knowledge_base = [item for item in loaded_data if item.get("embedding") and isinstance(item["embedding"], list)]
        for item in temp_knowledge_base:
            item["embedding"] = np.array(item["embedding"], dtype=float)
        knowledge_base_items = temp_knowledge_base
        logger.info(f"Successfully loaded {len(knowledge_base_items)} items.")
    except Exception as e:
        logger.error(f"Error loading knowledge base: {e}")
load_precomputed_knowledge()

def cosine_similarity(vec1, vec2):
    # (function is unchanged)
    if vec1 is None or vec2 is None: return 0.0
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0: return 0.0
    return dot_product / (norm_vec1 * norm_vec2)


@ai_bp.route("/api/ask", methods=["POST"])
@login_required
def ask_ai():
    logger.info(f"User '{current_user.username}' is making a request to /api/ask.")
    if not client:
        return jsonify({"error": "OpenAI client not initialized."}), 500
    
    data = request.get_json()
    if not data or "prompt" not in data or not data["prompt"].strip():
        return jsonify({"error": "Prompt cannot be empty"}), 400
        
    user_prompt = data["prompt"].strip().lower() # Convert to lowercase for easier keyword matching
    service_context = data.get("service_context")

    # --- UPDATED LOGIC: Handle the new simulation commands ---
    if service_context == 'digital_twin':
        # Simple keyword-based intent recognition
        if any(keyword in user_prompt for keyword in ['increase', 'faster', 'colder', 'speed up']):
            return jsonify({'action': 'increase_fan_speed', 'message': 'Increasing fan speed.'})
        elif any(keyword in user_prompt for keyword in ['decrease', 'slower', 'warmer', 'slow down']):
            return jsonify({'action': 'decrease_fan_speed', 'message': 'Decreasing fan speed.'})
        else:
            return jsonify({'action': 'none', 'message': "I can control the fan speed. Try 'make it colder'."})
    
    # --- The existing Business Consultant logic remains the same ---
    else:
        # (This entire section for consultant chat is unchanged)
        session = Session()
        try:
            cached_result = session.query(AICache).filter_by(prompt_text=user_prompt).first()
            if cached_result:
                logger.info("Serving response from cache.")
                # For non-streaming, just return the cached response. Adapt if you want to stream from cache.
                return jsonify({"response": cached_result.response_text, "source": "cache"})
        finally:
            session.close()

        # ... RAG logic remains ...
        user_prompt_embedding = get_embedding(user_prompt)
        relevant_chunks_text = []
        if user_prompt_embedding and knowledge_base_items:
            similarities = [{"text": item["text"], "source": item.get("source"), "score": cosine_similarity(user_prompt_embedding, item["embedding"])} for item in knowledge_base_items]
            similarities.sort(key=lambda x: x["score"], reverse=True)
            relevant_chunks_text = [f"- From {c['source']}: {c['text']}" for c in similarities[:3] if c['score'] > 0.6]
        
        retrieved_context_str = "\n\n".join(relevant_chunks_text) if relevant_chunks_text else "No relevant context found."
        system_message_content = "You are a general AI Business Advisor..."

        messages = [{"role": "system", "content": f"{system_message_content}\n\nRetrieved Context:\n{retrieved_context_str}"}, {"role": "user", "content": user_prompt}]
        
        def generate_response_chunks():
            # ... streaming and caching logic remains the same ...
            full_response = []
            try:
                stream = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, stream=True)
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response.append(content)
                        yield f"data: {json.dumps({'delta': content})}\n\n"
                
                final_text = "".join(full_response)
                if final_text:
                    cache_session = Session()
                    try:
                        new_entry = AICache(prompt_text=user_prompt, response_text=final_text)
                        cache_session.add(new_entry)
                        cache_session.commit()
                    except:
                        cache_session.rollback()
                    finally:
                        cache_session.close()
                
                yield f"data: {json.dumps({'event': 'done'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(generate_response_chunks(), mimetype='text/event-stream')


# --- REMOVED: The old temperature stream is no longer needed as it's simulated on the front-end ---
# The @ai_bp.route('/api/temperature') function can be safely deleted from this file.