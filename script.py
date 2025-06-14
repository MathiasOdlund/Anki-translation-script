#!/usr/bin/env python3
import os
import time
import base64
import logging
import requests
from transformers import pipeline
from PIL import Image
import numpy as np

# Optional: use GPU or MPS on Mac M1/M2/M4 via PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Enable/disable torch.compile on CUDA only
USE_COMPILE = True

def get_compute_settings():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda", torch.float16
    elif TORCH_AVAILABLE and torch.backends.mps.is_available():
        # bfloat16 support on newer MPS backends
        return "mps", torch.bfloat16
    else:
        return "cpu", None

DEVICE, TORCH_DTYPE = get_compute_settings()

# Configuration
ANKI_CONNECT_URL = "http://localhost:8765"
#DECK_NAME = "Polish-English"
DECK_NAME = "D1PolishEnglish"
TAG        = "d1"                # ← only process cards with this tag
POLISH_FIELD     = "Polish original"
TRANSLATION_FIELD= "Translation"
AUDIO_CHECK      = "[sound:"
IMAGE_CHECK      = "<img"
TTS_LANG         = "pl"
TTS_DELAY        = 0.5
LOG_LEVEL        = logging.DEBUG
MAX_IMG_ATTEMPTS = 3
INFERENCE_STEPS  = 20
IMAGE_MODEL_ID   = "stabilityai/stable-diffusion-2-1"

# Logging setup
logger = logging.getLogger(__name__)
def setup_logging():
    handler = logging.StreamHandler()
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(LOG_LEVEL)

# Persistent HTTP session for AnkiConnect
SESSION = requests.Session()

def invoke(action, params=None):
    payload = {"action": action, "version": 6, "params": params or {}}
    resp = SESSION.post(ANKI_CONNECT_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"AnkiConnect error: {data['error']}")
    return data["result"]

# Translation pipeline
def load_translator():
    logger.info("Loading translation pipeline...")
    return pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-pl-en",
        device=0 if DEVICE=="cuda" else -1,
        torch_dtype=TORCH_DTYPE
    )

translator = None
def translate_text(polish: str) -> str:
    global translator
    if translator is None:
        translator = load_translator()
    start = time.time()
    result = translator(polish, max_length=512)
    translation = result[0]["translation_text"]
    logger.debug(f"Translation result: {translation!r}")
    logger.info(f"Translation took {time.time() - start:.2f}s")
    return translation

# TTS generator
from gtts import gTTS
def generate_tts(text: str, lang: str, prefix: str, note_id: int) -> str:
    start = time.time()
    filename = f"{prefix}_{note_id}.mp3"
    gTTS(text=text, lang=lang).save(filename)
    logger.info(f"TTS saved '{filename}' in {time.time() - start:.2f}s")
    return filename

# Load and optimize Stable Diffusion pipeline
IMAGE_PIPELINE = None
def load_image_pipeline():
    global IMAGE_PIPELINE
    try:
        from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

        logger.info("Loading Stable Diffusion pipeline...")
        pipe = StableDiffusionPipeline.from_pretrained(
            IMAGE_MODEL_ID, torch_dtype=TORCH_DTYPE
        ).to(DEVICE)

        pipe.enable_attention_slicing()

        # Compile only on CUDA and if enabled
        if USE_COMPILE and DEVICE == "cuda":
            try:
                pipe.unet = torch.compile(pipe.unet)
                pipe.vae.decode = torch.compile(pipe.vae.decode)
                logger.info("Applied torch.compile to UNet and VAE")
            except Exception as e:
                logger.warning(f"torch.compile failed, falling back: {e}")

        # MPS-friendly layout
        pipe.unet.to(memory_format=torch.channels_last)
        pipe.vae.to(memory_format=torch.channels_last)
        try:
            pipe.unet.fuse_qkv_projections()
        except Exception:
            logger.debug("QKV fusion unavailable, continuing without it")

        # Faster scheduler
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

        # Disable safety checker
        def _no_safety(images, **kwargs):
            return images, [False] * len(images)
        pipe.safety_checker = _no_safety

        IMAGE_PIPELINE = pipe
        logger.info("Stable Diffusion pipeline ready.")
    except Exception as e:
        IMAGE_PIPELINE = None
        logger.warning(f"SD pipeline init failed: {e}. Image generation disabled.")

def generate_image(prompt: str, prefix: str, note_id: int) -> str:
    if IMAGE_PIPELINE is None:
        load_image_pipeline()
    if IMAGE_PIPELINE is None:
        raise RuntimeError("Image pipeline not initialized.")
    filename = f"{prefix}_{note_id}.png"
    for attempt in range(1, MAX_IMG_ATTEMPTS + 1):
        logger.debug(f"[Note {note_id}] Image attempt {attempt}")
        gen = None
        if TORCH_AVAILABLE and DEVICE=="cuda":
            gen = torch.Generator(device=DEVICE).manual_seed(note_id + attempt)
        start = time.time()
        out = IMAGE_PIPELINE(
            prompt,
            guidance_scale=7.5,
            num_inference_steps=INFERENCE_STEPS,
            generator=gen
        )
        img = out.images[0]
        img.save(filename)
        gray = img.convert("L")
        mean_val = np.array(gray).mean()
        if mean_val < 5:
            logger.warning(f"Blank image (mean {mean_val:.2f}); retrying...")
            os.remove(filename)
            continue
        logger.info(f"Image saved '{filename}' in {time.time() - start:.2f}s")
        return filename
    raise RuntimeError("All image attempts blank")

def upload_media(filename: str):
    start = time.time()
    with open(filename, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    invoke("storeMediaFile", {"filename": filename, "data": data})
    logger.info(f"Uploaded '{filename}' in {time.time() - start:.2f}s")

def main():
    setup_logging()
    logger.info(f"=== Starting Anki enrichment for deck='{DECK_NAME}', tag='{TAG}' ===")

    # Only fetch notes in the deck that also have our TAG
    query = f"deck:{DECK_NAME} tag:{TAG}"
    note_ids = invoke("findNotes", {"query": query})
    total = len(note_ids)
    if total == 0:
        logger.warning(f"No notes found for query: {query!r}")
        return

    decks = invoke("deckNames")
    if DECK_NAME not in decks:
        logger.error(f"Deck '{DECK_NAME}' not found; aborting.")
        return

    for idx, nid in enumerate(note_ids, 1):
        logger.info(f"[{idx}/{total}] Note ID {nid}")
        note = invoke("notesInfo", {"notes": [nid]})[0]
        pl = note["fields"][POLISH_FIELD]["value"]
        en = note["fields"][TRANSLATION_FIELD]["value"]
        tags = note.get("tags", [])

        # Sanity check: ensure TAG is actually on this note
        if TAG not in tags:
            logger.debug(f"Skipping Note {nid} (missing tag '{TAG}')")
            continue

        # skip if already has both
        if AUDIO_CHECK in pl and IMAGE_CHECK in pl:
            logger.debug("Skipping (already has audio & image)")
            continue

        # translation
        if not en.strip():
            en = translate_text(pl)

        # tts
        tts_file = None
        if AUDIO_CHECK not in pl:
            tts_file = generate_tts(pl, TTS_LANG, "pl", nid)
            upload_media(tts_file)

        # image
        img_file = None
        prompt = f"Illustrate the word '{en}'. Concrete nouns literal; abstract terms creative."
        logger.debug(f"Image prompt: {prompt!r}")
        try:
            img_file = generate_image(prompt, "img", nid)
            upload_media(img_file)
        except Exception as e:
            logger.error(f"Image generation skipped: {e}")

        # update note
        new_html = pl
        if tts_file:
            new_html += f"<br>[sound:{tts_file}]"
        if img_file:
            new_html += f"<br><img src=\"{img_file}\">"

        invoke("updateNoteFields", {
            "note": {
                "id": nid,
                "fields": {
                    POLISH_FIELD: new_html,
                    TRANSLATION_FIELD: en
                }
            }
        })
        logger.debug("Note fields updated")

        # cleanup
        for f in (tts_file, img_file):
            if f and os.path.exists(f):
                os.remove(f)

        time.sleep(TTS_DELAY)

    logger.info("=== Finished processing all notes ===")

if __name__ == "__main__":
    main()
