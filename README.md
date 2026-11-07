# SpokenAI

**An AI speech coach that scores how you think, speak and feel — in three rounds.**

SpokenAI is a Flask web app that helps people improve public speaking. Users record answers in the browser; the app transcribes them with Whisper and evaluates timing, relevance, creativity, coherence, fluency and vocal emotion, then gives per-round and overall feedback.

![Python](https://img.shields.io/badge/Python-3670A0?logo=python&logoColor=ffdd54) ![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white) ![Whisper](https://img.shields.io/badge/OpenAI_Whisper-412991?logo=openai&logoColor=white) ![Hugging Face](https://img.shields.io/badge/Transformers-FFD21E?logo=huggingface&logoColor=black) ![spaCy](https://img.shields.io/badge/spaCy-09A3D5?logo=spacy&logoColor=white)

---

## The three rounds

| Round | Task | What's evaluated |
|---|---|---|
| **1 · Rapid Fire** | Complete analogies within 5 seconds | Response time, correctness, creativity (semantic + fuzzy match) |
| **2 · Triple Step** | Speak for 5 minutes on a random topic while distractor words appear | Topic adherence, coherence, fluency, pauses / continuity |
| **3 · Conductor** | Read a sentence in an assigned emotion (angry, calm, excited, …) | Vocal emotion match from pitch, energy and tempo |

A final **overall feedback** page combines all rounds.

## How it works

- **Speech-to-text** — OpenAI Whisper and Distil-Whisper (Hugging Face pipeline) on normalised audio (pydub + ffmpeg)
- **Semantic scoring** — Sentence-Transformers embeddings + cosine similarity for relevance and topic adherence
- **Language analysis** — spaCy and textstat for coherence and readability
- **Fluency** — silence detection to measure pauses and continuity
- **Emotion** — librosa pitch (YIN), energy (RMS) and tempo matched against per-emotion vocal profiles (excited, calm, angry, sad, confident, …)
- **Accounts** — Flask-Login + SQLAlchemy (SQLite) to store users and scores

## Project Structure

```
SpokenAI/
├── __init__.py        # Flask app, auth, routes for all three rounds
├── utils.py           # Audio processing, transcription and scoring functions
├── templates/         # Instruction, round and feedback pages
└── requirements.txt
```

## Run it

**Prerequisites:** Python 3.10, [ffmpeg](https://ffmpeg.org/) on your PATH.

```bash
git clone https://github.com/kaviya-ns/SpokenAI.git
cd SpokenAI
pip install -r requirements.txt
echo SECRET_KEY=change-me > .env
flask --app __init__ run
```

Open <http://127.0.0.1:5000>, register, and start with Round 1.
