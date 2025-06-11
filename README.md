# Anki Polish TTS Uploader

A command-line Python utility to:

* Translate Polish → English offline using MarianMT
* Generate Polish-only pronunciation audio via gTTS
* Upload the audio to your Anki cards via AnkiConnect, skipping cards that already have audio

---

## Repository Structure

```text
├── script.py        # Main script
├── README.md        # This file
├── requirements.txt # Python dependencies
└── LICENSE          # MIT license
```

---

## Prerequisites

1. **Anki** installed, with the [AnkiConnect add-on (2055492159)](https://ankiweb.net/shared/info/2055492159) enabled.
2. **Python 3.8+** on your system.
3. (Optional) **git** if you want to clone from GitHub.

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/anki-polish-tts.git
   cd anki-polish-tts
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate    # macOS/Linux
   # .\.venv\Scripts\activate # Windows PowerShell
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

Open `script.py` and edit the top section to match your Anki setup:

```python
ANKI_CONNECT_URL = "http://localhost:8765"
DECK_NAME        = "Polish-English"      # exact deck name in Anki
POLISH_FIELD     = "Polish original"
EN_FIELD         = "Translation"
```

Make sure `DECK_NAME` matches exactly (including hyphens, spaces, capitalization).

---

## Usage

1. **Start Anki** so that AnkiConnect is listening.

2. **Activate** your virtual environment, if using one.

3. **Run** the script:

   ```bash
   python script.py
   ```

   You will see logs indicating:

    * Available decks
    * Number of notes found
    * Cards skipped (already have audio)
    * Cards updated with new audio and translation

4. **Stop & resume** anytime—the script skips cards that already contain `[sound:…]`.

---

## Adding Support for Other Languages

You can easily extend this utility to handle *any* language pair supported by MarianMT and gTTS (or Argos Translate). Follow these steps:

1. **Install or verify model availability**

   * **MarianMT**: pick the appropriate model from Hugging Face, e.g. `Helsinki-NLP/opus-mt-<SRC>-<TGT>`.
   * **Argos Translate**: ensure the package for `<SRC>→<TGT>` is available and installed via `argostranslate.package`.

2. **Update configuration variables** at the top of `script.py`:

   ```python
   # Language codes
   SRC_LANG      = "pl"             # original text language (e.g. "de", "es", "fr")
   TGT_LANG      = "en"             # translation text language
   TTS_LANG      = "pl"             # TTS language for audio output

   # Model identifiers
   MARIAN_MODEL  = "Helsinki-NLP/opus-mt-pl-en"
   # or for Argos Translate, update install_argos_model(from_code=SRC_LANG, to_code=TGT_LANG)
   ```

3. **Adjust the translator and TTS setup**:

   ```python
   # MarianMT pipeline
   translator = pipeline(
       "translation",
       model=MARIAN_MODEL,
       device=device
   )

   # gTTS uses TTS_LANG automatically
   ```

4. **Change field names if needed**

   * If you want to store your translated text in a different note field (e.g. `German↔English`), update:

     ```python
     POLISH_FIELD     = "SourceText"
     EN_FIELD          = "TranslatedText"
     ```

5. **Run the script**

   ```bash
   python script.py
   ```

   * The script will translate from `SRC_LANG` to `TGT_LANG` and generate TTS in `TTS_LANG`.
   * It will skip any cards that already contain `[sound:…]` in the source field.

6. **Verify in Anki**

   * Browse your deck and confirm audio icons appear on the source side.
   * Adjust any language-specific punctuation or markup if necessary.

---

With these minimal changes, you can support **any** language your study requires—just ensure the model and codes match your target pair!


## LICENSE

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
