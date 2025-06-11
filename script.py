import os
import base64
import logging
import time
import requests
from transformers import pipeline

# Optional: use GPU on Mac M1/M2 via PyTorch
try:
    import torch
    device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else -1)
except ImportError:
    device = -1

# Configuration
ANKI_CONNECT_URL = "http://localhost:8765"
DECK_NAME = "Polish-English"
POLISH_FIELD = "Polish original"             # Field name for Polish text
TRANSLATION_FIELD = "Translation"             # Field name for English translation text
AUDIO_CHECK = "[sound:"                       # Marker to identify existing audio
TTS_LANG = "pl"                              # Only Polish audio
TTS_DELAY = 0.5                                # seconds between TTS API calls to avoid rate limits
LOG_LEVEL = logging.INFO                      # Set to DEBUG for verbose logs

# Setup translation pipeline (Helsinki-NLP MarianMT)
translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-pl-en",
    device=device
)

# Setup logging
def setup_logging():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

# Configuration
ANKI_CONNECT_URL = "http://localhost:8765"
POLISH_FIELD = "Polish original"             # Field name for Polish text
ENGLISH_FIELD = "Translation"                # Field name for English text
AUDIO_FIELD = None                             # Optional: set to a field name to store audio there
TTS_DELAY = 0.5                                # seconds between API calls to avoid rate limits
LOG_LEVEL = logging.INFO                      # Set to DEBUG for more verbose logging

# Setup logging
def setup_logging():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

# Invoke AnkiConnect actions
def invoke(action, params=None):
    payload = {"action": action, "version": 6}
    if params:
        payload["params"] = params
    try:
        response = requests.post(ANKI_CONNECT_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        if result.get("error"):
            raise Exception(f"AnkiConnect error: {result['error']}")
        return result.get("result")
    except Exception:
        logging.exception(f"Failed to invoke '{action}' with params {params}")
        raise

# 1. Translate Polish text to English if field is empty or outdated
def translate_text(polish: str) -> str:
    try:
        result = translator(polish, max_length=512)
        translation = result[0]["translation_text"]
        logging.info(f"Generated translation: {translation}")
        return translation
    except Exception:
        logging.exception("Error during translation")
        raise

# 2. Generate TTS audio and return filename (only Polish)
def generate_tts(text: str, lang: str, prefix: str, note_id: int) -> str:
    safe_prefix = f"{prefix}_{note_id}"
    filename = f"{safe_prefix}.mp3"
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang)
        tts.save(filename)
        logging.info(f"Saved TTS file '{filename}' for lang='{lang}'")
        return filename
    except Exception:
        logging.exception(f"Error generating TTS for note {note_id} ({lang})")
        raise

# 3. Upload file to Anki media collection
def upload_media(filename: str):
    try:
        with open(filename, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        invoke("storeMediaFile", {"filename": filename, "data": data})
        logging.info(f"Uploaded '{filename}' to Anki media collection")
    except Exception:
        logging.exception(f"Error uploading media file '{filename}'")
        raise
def upload_media(filename: str):
    try:
        with open(filename, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        invoke("storeMediaFile", {"filename": filename, "data": data})
        logging.info(f"Uploaded '{filename}' to Anki media collection")
    except Exception:
        logging.exception(f"Error uploading media file '{filename}'")
        raise

# Main workflow
def main():
    setup_logging()
    logging.info("Starting audio generation and upload process")

    # 1. Find all note IDs in the deck
    note_ids = invoke("findNotes", {"query": f"deck:{DECK_NAME}"})
    decks = invoke("deckNames")
    logging.info(f"Available decks: {decks}")
    if DECK_NAME not in decks:
        logging.error(f"Deck '{DECK_NAME}' not found! Please set DECK_NAME to one of the above.")
        return
    total = len(note_ids)
    logging.info(f"Found {total} notes in deck '{DECK_NAME}'")

    for idx, nid in enumerate(note_ids, start=1):
        logging.info(f"Processing note {idx}/{total} (ID={nid})")
        try:
            note = invoke("notesInfo", {"notes": [nid]})[0]
            fields = note["fields"]

            polish_text = fields[POLISH_FIELD]["value"]
            current_translation = fields[TRANSLATION_FIELD]["value"]

            # Skip cards that already have Polish audio
            if AUDIO_CHECK in polish_text:
                logging.info(f"Skipping note {nid}: audio already present")
                continue

            # Translate if translation field is empty
            if not current_translation.strip():
                new_translation = translate_text(polish_text)
            else:
                new_translation = current_translation

            # Generate Polish audio
            pl_file = generate_tts(polish_text, lang=TTS_LANG, prefix="pl", note_id=nid)

            # Upload to Anki
            upload_media(pl_file)

            # Prepare updated fields: append audio to Polish field, update translation field
            updated_fields = {
                POLISH_FIELD: polish_text + f"<br>[sound:{pl_file}]",
                TRANSLATION_FIELD: new_translation
            }

            # Update the note
            invoke("updateNoteFields", {"note": {"id": nid, "fields": updated_fields}})
            logging.info(f"Updated note {nid} with audio and translation")

            # Cleanup local files
            try:
                os.remove(pl_file)
                logging.debug(f"Deleted local file '{pl_file}'")
            except OSError:
                logging.warning(f"Could not delete local file '{pl_file}'")

            time.sleep(TTS_DELAY)

        except Exception:
            logging.error(f"Skipping note ID={nid} due to errors")
            continue
            for f in (pl_file, en_file):
                try:
                    os.remove(f)
                    logging.debug(f"Deleted local file '{f}'")
                except OSError:
                    logging.warning(f"Could not delete local file '{f}'")

            time.sleep(TTS_DELAY)

        except Exception:
            logging.error(f"Skipping note ID={nid} due to errors")
            continue

    logging.info("All notes processed. Exiting.")

if __name__ == "__main__":
    setup_logging()
    main()

if __name__ == "__main__":
    main()
