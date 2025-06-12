import os
import base64
import logging
import time
import requests
from transformers import pipeline

# Optional: use GPU or MPS on Mac M1/M2/M4 via PyTorch
torch_available = False
try:
    import torch
    torch_available = True
    device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else -1)
except ImportError:
    device = -1

# For image inspection
from PIL import Image
import numpy as np

# Configuration constants
ANKI_CONNECT_URL = "http://localhost:8765"
DECK_NAME = "D1PolishEnglish"
POLISH_FIELD = "Polish original"
TRANSLATION_FIELD = "Translation"
AUDIO_CHECK = "[sound:"
IMAGE_CHECK = "<img"
TTS_LANG = "pl"
TTS_DELAY = 0.5
LOG_LEVEL = logging.INFO
MAX_IMG_ATTEMPTS = 3        # retries for non-blank image
INFERENCE_STEPS = 30        # diffusion steps per image
IMAGE_MODEL_ID = "stabilityai/stable-diffusion-2-1"

# Initialize logging

def setup_logging():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

# AnkiConnect invocation helper

def invoke(action, params=None):
    payload = {"action": action, "version": 6}
    if params:
        payload["params"] = params
    response = requests.post(ANKI_CONNECT_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise Exception(f"AnkiConnect error: {data['error']}")
    return data.get("result")

# Load translation pipeline
er = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-pl-en",
    device=device
)

def translate_text(polish: str) -> str:
    result = er(polish, max_length=512)
    translation = result[0]["translation_text"]
    logging.info(f"Generated translation: {translation}")
    return translation

# TTS generator
def generate_tts(text: str, lang: str, prefix: str, note_id: int) -> str:
    filename = f"{prefix}_{note_id}.mp3"
    from gtts import gTTS
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)
    logging.info(f"Saved TTS file '{filename}'")
    return filename

# Diffusion-based image generator with retries (original SD v1.5)
try:
    from diffusers import StableDiffusionPipeline
    # Load the original runwayml v1.5 model
    IMAGE_PIPELINE = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=(torch.float16 if torch_available else None)
    ).to(device)
    IMAGE_PIPELINE.enable_attention_slicing()
    # Override safety checker to bypass NSFW filtering
    def noop_safety(images, **kwargs):
        return images, [False] * len(images)
    IMAGE_PIPELINE.safety_checker = noop_safety
    logging.info("Loaded SD v1.5 pipeline with custom safety checker.")
except Exception as e:
    IMAGE_PIPELINE = None
    logging.warning(f"Image pipeline init failed: {e}. Image generation disabled.")

# Generate a non-blank image
def generate_image(prompt: str, prefix: str, note_id: int) -> str:
    if IMAGE_PIPELINE is None:
        raise RuntimeError("Image pipeline not initialized.")
    filename = f"{prefix}_{note_id}.png"
    for attempt in range(1, MAX_IMG_ATTEMPTS + 1):
        logging.info(f"Image gen attempt {attempt}/{MAX_IMG_ATTEMPTS} for note {note_id}")
        gen = None
        if torch_available:
            gen = torch.Generator(device=device).manual_seed(note_id + attempt)
        out = IMAGE_PIPELINE(
            prompt,
            guidance_scale=7.5,
            num_inference_steps=INFERENCE_STEPS,
            generator=gen
        )
        img = out.images[0]
        img.save(filename)
        gray = img.convert("L")
        arr = np.array(gray)
        if arr.mean() < 5:
            logging.warning(f"Blank image (mean {arr.mean():.2f}); retrying...")
            os.remove(filename)
            continue
        logging.info(f"Valid image saved '{filename}'")
        return filename
    raise RuntimeError(
        f"All {MAX_IMG_ATTEMPTS} image attempts blank for note {note_id}")

# Upload media file to Anki

def upload_media(filename: str):
    with open(filename, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    invoke("storeMediaFile", {"filename": filename, "data": data})
    logging.info(f"Uploaded '{filename}'")

# Main processing loop
def main():
    setup_logging()
    logging.info("Starting Anki enrichment")

    note_ids = invoke("findNotes", {"query": f"deck:{DECK_NAME}"})
    total = len(note_ids)
    decks = invoke("deckNames")
    if DECK_NAME not in decks:
        logging.error(f"Deck '{DECK_NAME}' missing")
        return

    for idx, nid in enumerate(note_ids, 1):
        logging.info(f"[{idx}/{total}] Note ID={nid}")
        note = invoke("notesInfo", {"notes": [nid]})[0]
        pl = note["fields"][POLISH_FIELD]["value"]
        en = note["fields"][TRANSLATION_FIELD]["value"]

        # skip if both exist
        if AUDIO_CHECK in pl and IMAGE_CHECK in pl:
            logging.info("Skip: audio+image present")
            continue

        # translate if needed
        if not en.strip():
            en = translate_text(pl)

        # TTS if missing
        if AUDIO_CHECK not in pl:
            tts = generate_tts(pl, TTS_LANG, "pl", nid)
            upload_media(tts)
        else:
            tts = None

        # image generation
        prompt = f"Illustrate the word '{en}'. Concrete nouns literal; abstract terms creative."
        logging.info(f"Prompt: {prompt}")
        img_file = None
        try:
            img_file = generate_image(prompt, "img", nid)
            upload_media(img_file)
        except Exception as e:
            logging.error(f"Image skipped: {e}")

        # assemble update
        new_pl = pl
        if tts: new_pl += f"<br>[sound:{tts}]"
        if img_file: new_pl += f"<br><img src=\"{img_file}\">"

        invoke("updateNoteFields", {"note": {"id": nid, "fields": {
            POLISH_FIELD: new_pl,
            TRANSLATION_FIELD: en
        }}})

        # cleanup
        for f in (tts, img_file):
            if f and os.path.exists(f): os.remove(f)
        time.sleep(TTS_DELAY)

    logging.info("Finished processing all notes.")

if __name__ == "__main__":
    main()
