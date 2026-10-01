"""
Avó's Fridge Rescue — Flask API
Serves whichever grandma persona currently holds the MLflow Prompt Registry
@champion alias, and turns a list of fridge ingredients into a recipe.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import mlflow
import mlflow.genai
import logging
import os
import threading
import uuid
from collections import OrderedDict
from datetime import datetime
from typing import Dict, Any, List

from src import llm_client, grandma_personas

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # Lovable's frontend calls this API from a different origin

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')

CHAT_EXPERIMENT_NAME = os.getenv('MLFLOW_CHAT_EXPERIMENT', 'Avo-Fridge-Live')

MAX_HISTORY_TURNS = int(os.getenv('CHAT_HISTORY_TURNS', 6))
MAX_CONVERSATIONS = int(os.getenv('CHAT_MAX_CONVERSATIONS', 200))
MAX_FRIDGE_ITEMS_LENGTH = 500

MODEL_NAME = llm_client.DEFAULT_MODEL
MAX_TOKENS = llm_client.DEFAULT_MAX_TOKENS

PROMPT_NAME = grandma_personas.PROMPT_NAME
AVO_PERSONAS = grandma_personas.PERSONAS

current_persona: Dict[str, Any] = {}
persona_info: Dict[str, Any] = {}
tracing_enabled = False

_conversations: "OrderedDict[str, List[Dict[str, str]]]" = OrderedDict()
_conversations_lock = threading.Lock()


def setup_tracing() -> bool:
    """Turn on MLflow tracing for every LLM call this service makes."""
    global tracing_enabled
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(CHAT_EXPERIMENT_NAME)
        mlflow.openai.autolog()
        tracing_enabled = True
        logger.info(f"Tracing into MLflow experiment '{CHAT_EXPERIMENT_NAME}'")
    except Exception as e:
        tracing_enabled = False
        logger.warning(f"Tracing is off — could not reach MLflow at {MLFLOW_TRACKING_URI}: {e}")
    return tracing_enabled


def get_history(session_id: str) -> List[Dict[str, str]]:
    with _conversations_lock:
        return list(_conversations.get(session_id, []))


def remember_turn(session_id: str, question: str, answer: str) -> None:
    with _conversations_lock:
        history = _conversations.pop(session_id, [])
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        _conversations[session_id] = history[-(MAX_HISTORY_TURNS * 2):]
        while len(_conversations) > MAX_CONVERSATIONS:
            _conversations.popitem(last=False)


def use_persona(persona_name: str) -> bool:
    """Serve one of the locally defined personas (fallback, before anything is promoted)."""
    global current_persona, persona_info

    if persona_name not in AVO_PERSONAS:
        logger.warning(f"Unknown persona: {persona_name}")
        return False

    persona = AVO_PERSONAS[persona_name]
    current_persona = {
        "name": persona_name,
        "template": persona["template"],
        "description": persona["description"],
    }
    persona_info = {
        "prompt_name": PROMPT_NAME,
        "persona": persona_name,
        "version": None,
        "source": "local",
        "status": "local_fallback",
        "description": persona["description"],
        "loaded_at": datetime.now().isoformat(),
    }
    logger.info(f"Serving local persona: {persona_name}")
    return True


def load_champion_persona() -> bool:
    """
    Serve whichever persona currently carries the @champion alias.

    Falls back to the local 'affectionate' persona when no champion exists yet.
    """
    global current_persona, persona_info

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@champion")

        persona_name = (prompt.tags or {}).get("persona", "unknown")
        current_persona = {
            "name": persona_name,
            "template": prompt.template,
            "description": f"champion, version {prompt.version}",
        }
        persona_info = {
            "prompt_name": PROMPT_NAME,
            "persona": persona_name,
            "version": prompt.version,
            "source": "mlflow",
            "status": "champion",
            "description": f"prompts:/{PROMPT_NAME}@champion",
            "loaded_at": datetime.now().isoformat(),
        }
        logger.info(f"Serving @champion: {PROMPT_NAME} v{prompt.version} ({persona_name})")
        return True

    except Exception as e:
        logger.warning(f"No @champion in the registry yet ({e}); falling back to local 'affectionate'")
        return use_persona("affectionate")


@mlflow.trace(span_type="LLM")
def query_llm(prompt: str, history: List[Dict[str, str]] | None = None):
    """
    Ask Avó's configured LLM (Gemini, or Ollama on fallback) and return the answer
    plus which provider actually produced it.
    """
    messages = list(history or [])
    messages.append({"role": "user", "content": prompt})

    try:
        response, provider = llm_client.generate(messages, max_tokens=MAX_TOKENS, temperature=0.7)

        if not response.choices:
            logger.error("LLM returned no choices")
            return "Sorry, I couldn't generate a recipe right now.", provider

        choice = response.choices[0]
        if choice.finish_reason == "length":
            logger.warning("Reply hit the token limit (max_tokens=%s) and was cut off", MAX_TOKENS)

        return choice.message.content or "Sorry, I couldn't generate a recipe right now.", provider

    except Exception as e:
        logger.error(f"Error querying the LLM: {str(e)}")
        return "Sorry, I ran into a problem in the kitchen. Please try again.", "none"


@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('index.html')
    # return jsonify({
    #     "service": "avo-fridge-api",
    #     "message": "Avó is listening. POST fridge_items to /recipe.",
    #     "endpoints": ["/health", "/recipe", "/persona/switch", "/persona/reload", "/persona/info"],
    # })


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "avo-fridge-api",
        "bot_name": "Avó",
        "persona_info": persona_info,
        "llm": llm_client.describe(),
        "tracing": {
            "enabled": tracing_enabled,
            "experiment": CHAT_EXPERIMENT_NAME,
            "tracking_uri": MLFLOW_TRACKING_URI,
            "active_sessions": len(_conversations),
            "history_turns": MAX_HISTORY_TURNS,
        },
    })


@mlflow.trace(name="fridge_recipe_turn", span_type="CHAIN")
def make_recipe(fridge_items: str, session_id: str, turn: int) -> Dict[str, Any]:
    """
    One turn: render the persona's prompt with the fridge contents, ask the model,
    remember the answer, and tag the trace so it's readable in MLflow.
    """
    load_champion_persona()
    prompt = grandma_personas.render(current_persona["name"], fridge_items)
    history = get_history(session_id)

    formatted_prompt = current_persona['template'].replace('{{fridge_items}}', fridge_items)

    ai_response, provider = query_llm(formatted_prompt, history=history)

    mlflow.update_current_trace(
        session_id=session_id,
        tags={
            "persona": current_persona['name'],
            "prompt_source": persona_info.get("source", "unknown"),
            "prompt_version": str(persona_info.get("version", "local")),
            "provider": provider,
            "turn": str(turn),
        },
    )

    remember_turn(session_id, fridge_items, ai_response)
    return {"response": ai_response, "provider": provider}


@app.route('/recipe', methods=['POST'])
def recipe():
    """Turn a list of fridge ingredients into a recipe, in Avó's current persona."""
    try:
        data = request.get_json()

        if not data or 'fridge_items' not in data:
            return jsonify({"error": "Missing 'fridge_items' field in request"}), 400

        fridge_items = data['fridge_items']

        if not fridge_items.strip():
            return jsonify({"error": "fridge_items cannot be empty"}), 400

        if len(fridge_items) > MAX_FRIDGE_ITEMS_LENGTH:
            return jsonify({"error": f"fridge_items too long (max {MAX_FRIDGE_ITEMS_LENGTH} characters)"}), 400

        session_id = (data.get('session_id') or '').strip() or f"session-{uuid.uuid4().hex[:12]}"
        turn = len(get_history(session_id)) // 2 + 1

        logger.info(f"Avó processing: {fridge_items[:50]}... [session {session_id} turn {turn}]")
        result = make_recipe(fridge_items, session_id, turn)

        return jsonify({
            "fridge_items": fridge_items,
            "response": result["response"],
            "provider": result["provider"],
            "timestamp": datetime.now().isoformat(),
            "bot_name": "Avó",
            "model": MODEL_NAME,
            "persona": current_persona.get('name'),
            "session_id": session_id,
            "turn": turn,
        })

    except Exception as e:
        logger.error(f"Error in /recipe: {str(e)}")
        return jsonify({
            "error": "Avó is having trouble right now. Please try again.",
            "timestamp": datetime.now().isoformat(),
        }), 500


@app.route('/persona/switch', methods=['POST'])
def switch_persona():
    """Force a specific persona, bypassing @champion (useful for demos/testing)."""
    try:
        data = request.get_json()
        persona_name = data.get('persona', 'affectionate')

        if use_persona(persona_name):
            return jsonify({
                "status": "success",
                "message": f"Switched to {persona_name}",
                "current_persona": current_persona,
                "persona_info": persona_info,
                "timestamp": datetime.now().isoformat(),
            })
        return jsonify({"error": "Unknown persona"}), 400
    except Exception as e:
        logger.error(f"Error switching persona: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/persona/reload', methods=['POST'])
def reload_persona_from_mlflow():
    """Re-read prompts:/avo-fridge-persona@champion. Call after promoting a new persona."""
    try:
        if load_champion_persona():
            return jsonify({
                "status": "success",
                "message": "Reloaded @champion from MLflow",
                "current_persona": current_persona,
                "persona_info": persona_info,
                "timestamp": datetime.now().isoformat(),
            })
        return jsonify({"error": "Failed to reload persona from MLflow"}), 400
    except Exception as e:
        logger.error(f"Error reloading persona: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/persona/info', methods=['GET'])
def persona_info_route():
    load_champion_persona
    return jsonify(persona_info)
    # return jsonify({
    #     "current_persona": current_persona,
    #     "available_personas": list(AVO_PERSONAS.keys()),
    #     "persona_info": persona_info,
    #     "timestamp": datetime.now().isoformat(),
    # })


def initialize_app():
    logger.info("Initializing Avó's Fridge Rescue API")
    setup_tracing()

    if not load_champion_persona():
        logger.warning("Using fallback configuration")

    if not llm_client.is_configured():
        logger.warning("No GEMINI_API_KEY set — Avó will start but cannot answer questions")
        logger.warning("   Put it in docker/.env, next to docker-compose.yml")
    else:
        logger.info(f"LLM client ready: {MODEL_NAME} (max_tokens={MAX_TOKENS})")

    logger.info("Avó's Fridge Rescue API initialized successfully")
    logger.info("Access the API at: http://localhost:8000")


if __name__ == '__main__':
    initialize_app()
    app.run(host='0.0.0.0', port=8000, debug=True)
