import os
import logging
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import pickle
import torch
from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pickled EasyOCR reader
def load_ocr_reader(pickle_path):
    try:
        with open(pickle_path, 'rb') as f:
            ocr_reader = pickle.load(f)
        return ocr_reader
    except Exception as e:
        logger.error(f"Failed to load OCR reader from pickle: {e}")
        raise

# Load translation models
def load_translation_models():
    models = {}
    tokenizers = {}
    language_pairs = [
         {"pair": "es-en", "repo_id": "dreyyyy/ES-EN"},
            {"pair": "en-es", "repo_id": "dreyyyy/EN-ES"},
            {"pair": "de-en", "repo_id": "dreyyyy/de-en"},
            {"pair": "de-es", "repo_id": "dreyyyy/de-es"},
            {"pair": "de-it", "repo_id": "dreyyyy/de-it"},
            {"pair": "de-tl", "repo_id": "dreyyyy/de-tl"},
            {"pair": "en-de", "repo_id": "dreyyyy/en-de"},
            {"pair": "en-it", "repo_id": "dreyyyy/en-it"},
            {"pair": "en-tl", "repo_id": "dreyyyy/en-tl"},
            {"pair": "es-de", "repo_id": "dreyyyy/es-de"},
            {"pair": "es-it", "repo_id": "dreyyyy/es-it"},
            {"pair": "es-tl", "repo_id": "dreyyyy/es-tl"},
            {"pair": "it-de", "repo_id": "dreyyyy/it-de"},
            {"pair": "it-en", "repo_id": "dreyyyy/it-en"},
            {"pair": "it-es", "repo_id": "dreyyyy/it-es"},
            {"pair": "tl-de", "repo_id": "dreyyyy/tl-de"},
            {"pair": "tl-en", "repo_id": "dreyyyy/tl-en"},
            {"pair": "tl-es", "repo_id": "dreyyyy/tl-es"},
    ]
    
    for entry in language_pairs:
        try:
            model_key = f"{entry['pair']}-model"
            models[model_key] = MarianMTModel.from_pretrained(entry["repo_id"])
            tokenizers[model_key] = MarianTokenizer.from_pretrained(entry["repo_id"])
        except Exception as e:
            logger.error(f"Failed to load model {entry['pair']}: {e}")
    
    return models, tokenizers

# Global variables
try:
    ocr_reader = load_ocr_reader('easyocr_reader.pkl')
    translation_models, translation_tokenizers = load_translation_models()
except Exception as e:
    logger.critical(f"Startup error: {e}")
    raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_image(image_path):
    try:
        results = ocr_reader.readtext(image_path)
        text = " ".join([result[1] for result in results])
        return text.strip()
    except Exception as e:
        logger.error(f"OCR extraction error: {e}")
        raise

def translate_text(source_lang, target_lang, text):
    model_key = f"{source_lang}-{target_lang}-model"
    if model_key not in translation_models:
        raise ValueError(f"Unsupported language pair: {model_key}")

    model = translation_models[model_key]
    tokenizer = translation_tokenizers[model_key]

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(inputs["input_ids"])
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    
    return translated_text

@app.route('/ocr_translate', methods=['POST'])
def ocr_translate():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file"}), 400

        file = request.files['image']
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        # Save and process image
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        # Extract text
        extracted_text = extract_text_from_image(image_path)
        if not extracted_text:
            return jsonify({"error": "No text found in image"}), 400

        # Get translation parameters
        source_lang = request.form.get('source_lang', 'en')
        target_lang = request.form.get('target_lang', 'es')

        # Translate
        translated_text = translate_text(source_lang, target_lang, extracted_text)

        # Clean up
        os.remove(image_path)

        return jsonify({
            "extracted_text": extracted_text,
            "translated_text": translated_text
        })

    except Exception as e:
        logger.error(f"OCR translation error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()
        text = data.get('text')
        source_lang = data.get('source_lang', 'en')
        target_lang = data.get('target_lang', 'es')

        translated_text = translate_text(source_lang, target_lang, text)
        return jsonify({"translated_text": translated_text})

    except Exception as e:
        logger.error(f"Text translation error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)