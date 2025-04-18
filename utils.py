import librosa
import numpy as np
from sentence_transformers import SentenceTransformer, util
from pydub import AudioSegment 
from pydub.silence import detect_silence 
import time
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import util
import os
from pydub import AudioSegment
from fuzzywuzzy import fuzz
import whisper
import wave
import random
import spacy
import textstat
import re
import scipy.signal
import scipy.signal.windows
import os
import subprocess
from transformers import pipeline

scipy.signal.hann = scipy.signal.windows.hann

# Path to the FFmpeg executable
ffmpeg_path = r"F:/ffmpeg-7.1-full_build/ffmpeg-7.1-full_build/bin"

whisper_model_base = whisper.load_model("base")
whisper_model_small = whisper.load_model("small")
distil_whisper_pipe = None
os.environ["PATH"] += os.pathsep +ffmpeg_path
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_md")  

def normalize_audio(input_path, output_path, target_dBFS=-20.0):
    try:
        # Load the audio file
        audio = AudioSegment.from_file(input_path)

        # Calculate the current dBFS level
        current_dBFS = audio.dBFS

        # Compute the required gain to normalize
        gain_needed = target_dBFS - current_dBFS

        # Apply gain to normalize the audio
        normalized_audio = audio.apply_gain(gain_needed)

        # Export the normalized audio to a WAV file
        normalized_audio.export(output_path, format="wav")
        print(f"✅ Normalized audio saved to: {output_path} (Adjusted by {gain_needed:.2f} dB)")
    except Exception as e:
        print(f"❌ Error normalizing audio: {e}")
        raise

def load_distil_whisper():
    global distil_whisper_pipe
    if distil_whisper_pipe is None:
        print("⚡ Loading Distil-Whisper...")
        distil_whisper_pipe = pipeline(
            "automatic-speech-recognition",
            "distil-whisper/distil-medium.en",
            device="cuda" if whisper_model_small.device.type == "cuda" else "cpu"
        )
        
# Example usage
def process_audio(input_webm,model):
    # Convert .webm to .wav
    wav_path = input_webm.replace(".webm", ".wav")
    convert_to_wav(input_webm, wav_path)

    # Normalize the audio
    normalized_wav_path = wav_path.replace(".wav", "_normalized.wav")
    normalize_audio(wav_path, normalized_wav_path)

    # Validate the normalized WAV file
    if is_valid_wav(normalized_wav_path):
        # Transcribe the audio
        if model=="base":
            transcription = transcribe_audio_base(normalized_wav_path)
        else:
            transcription = transcribe_audio_small(normalized_wav_path)
        if transcription:
            print(f"Transcription: {transcription}")
        else:
            print("Transcription failed.")
    else:
        print("Invalid WAV file.")

def transcribe_audio_base(audio_file):
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return None

    try:
        # 1. Verify audio file integrity
        if os.path.getsize(audio_file) < 2 * 1024:  # At least 2KB
            print(f"⚠️ Suspiciously small audio file: {os.path.getsize(audio_file)} bytes")
            return None

        # 2. Load with librosa for better error handling
        try:
            audio, sr = librosa.load(audio_file, sr=16000)
            if len(audio) < 1600:  # Less than 0.1s
                print(f"⚠️ Audio too short: {len(audio)/16000:.2f}s")
                return None
        except Exception as e:
            print(f"❌ Librosa load failed: {e}")
            return None

        # 3. Initialize model with error handling
        global whisper_model_base
        if 'whisper_model_base' not in globals():
            try:
                whisper_model_base = whisper.load_model("base")
            except Exception as e:
                print(f"❌ Model loading failed: {e}")
                return None

        # 4. Audio processing with safety checks
        try:
            audio = whisper.pad_or_trim(audio)
            if np.max(np.abs(audio)) < 0.01:
                print(f"⚠️ Silent audio (max amplitude: {np.max(np.abs(audio))})")
                return None

            mel = whisper.log_mel_spectrogram(audio)
            if mel.shape[-1] < 10:  # Minimum frames
                print(f"⚠️ Insufficient audio frames: {mel.shape}")
                return None
            mel = mel.to(whisper_model_base.device)
        except Exception as e:
            print(f"❌ Audio processing failed: {e}")
            return None

        # 5. Transcription with multiple fallbacks
        transcription = None
        for attempt in range(3):  # Try up to 3 times
            try:
                # Method 1: Standard decode
                options = whisper.DecodingOptions(
                    language="en",
                    fp16=whisper_model_base.device.type == "cuda",
                    without_timestamps=True
                )
                result = whisper.decode(whisper_model_base, mel, options)
                transcription = result.text.strip().lower()
                
                # Validate
                if transcription and transcription not in {"", "you"}:
                    break
                    
                # Method 2: Fallback to transcribe()
                if attempt == 1:
                    result = whisper_model_base.transcribe(audio_file)
                    transcription = result["text"].strip().lower()
                    if transcription: break
                    
                # Method 3: Alternative processing
                if attempt == 2:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                    audio = whisper.pad_or_trim(audio)
                    result = whisper_model_base.transcribe(audio)
                    transcription = result["text"].strip().lower()
                    
            except Exception as e:
                print(f"⚠️ Attempt {attempt+1} failed: {e}")
                time.sleep(0.5 * (attempt + 1))  # Backoff delay

        return transcription if transcription and transcription not in {"", "you"} else None

    except Exception as e:
        print(f"❌ Critical failure: {e}")
        return None
    
def transcribe_audio_small(audio_file, use_distil=False):
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return None
    
    try:
        # Audio preprocessing
        audio = whisper.load_audio(audio_file)
        
        if np.max(np.abs(audio)) < 0.01:
            print(f"⚠️ Audio is silent or too quiet: {audio_file}")
            return None
        
        # For Distil-Whisper, increase chunk size and ensure full processing
        load_distil_whisper()  # Ensure model is loaded
        
        # Use a larger chunk size for longer audio and ensure all chunks are processed
        result = distil_whisper_pipe(
            audio_file, 
            chunk_length_s=60,  # Increase chunk size
            stride_length_s=5,   # Add overlap between chunks
            batch_size=8         # Process more chunks at once if memory allows
        )
        transcription = result["text"].strip().lower()
        
        # Post-processing validation
        if not transcription or transcription in {"you", ""}:
            print(f"⚠️ Empty/Invalid transcription: {audio_file}")
            return None
        
        # Optional: add debugging to check transcription length
        print(f"Transcription length: {len(transcription.split())} words")
        
        return transcription
    
    except Exception as e:
        print(f"❌ Transcription failed: {str(e)}")
        return None

def convert_to_wav(input_path, output_path):
    try:
        audio = AudioSegment.from_file(input_path, format="webm")
        audio.export(output_path, format="wav")
        print(f"Successfully converted {input_path} to {output_path}")
    except Exception as e:
        print(f"Error converting {input_path} to WAV: {e}")
        raise

def is_valid_wav(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return False

    # Check if the file is not empty
    if os.path.getsize(file_path) == 0:
        print(f"File is empty: {file_path}")
        return False

    # Try opening the file with the wave module
    try:
        with wave.open(file_path, 'rb') as wav_file:
            # Check basic WAV file properties
            if wav_file.getnchannels() not in [1, 2]:  # Mono or stereo
                print(f"Invalid number of channels: {wav_file.getnchannels()}")
                return False
            if wav_file.getframerate() < 8000:  # Minimum sample rate
                print(f"Invalid sample rate: {wav_file.getframerate()}")
                return False

            # Check if the audio is silent
            frames = wav_file.readframes(wav_file.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)
            if np.max(np.abs(audio_data)) < 100:  # Threshold for silence
                print("Audio is silent or too quiet.")
                return False

            return True
    except wave.Error as e:
        print(f"Invalid WAV file: {e}")
        return False
    except EOFError as e:
        print(f"Unexpected end of file: {e}")
        return False
    
def timing(total_time):
    print(f"Total Time: {total_time}")
    if total_time is None or total_time < 0:
        return 0  # Invalid time
    elif total_time < 2:
        return 9  # Excellent response time
    elif 2<=total_time<3:
        return 8   # Good response time
    elif total_time>3 :
        return 3   # Average response time
    else:
        return 1   # Very slow response time


def relevance(user_ans, context):
    # Clean inputs
    user_ans = user_ans.strip().lower()
    context = context.strip().lower()
    
    # Early returns for invalid cases
    if not user_ans or not context:
        return 0.0
    if user_ans == "you":
        return 0.0
    
    # Exact match check (including partial matches)
    if user_ans == context:
        return 10.0
    if context in user_ans: 
        return 10.0
    if user_ans in context:  
        return 10.0
    
    # Semantic similarity fallback
    try:
        user_embedding = embedding_model.encode(user_ans, convert_to_tensor=True)
        context_embedding = embedding_model.encode(context, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(user_embedding, context_embedding).item()
        
        # Convert [-1,1] to [1,10] with non-linear scaling
        if similarity > 0.8:  # Very similar
            return min(9.5, round(9 * similarity + 1, 2))  # Cap near-perfect matches at 9.5
        else:
            return max(1.0, round(5 * similarity + 5, 2))  # Linear scaling for lower scores
    
    except Exception as e:
        print(f"⚠️ Relevance scoring error: {e}")
        return 0.0

def creativity(user_ans, correct_ans, question_context=None):
    # Validate inputs
    if not user_ans or not correct_ans:
        return 1  # Default minimum score
    
    user_ans = user_ans.lower().strip()
    if user_ans in {"", "you", "uh", "um", "none"}:
        return 1  # Default minimum for invalid inputs

    try:
        # Normalize correct answers to list
        correct_ans = [correct_ans] if isinstance(correct_ans, str) else correct_ans
        
        # Calculate relevance scores
        relevance_scores = []
        for ans in correct_ans:
            ans = ans.lower().strip()
            if question_context:
                score = relevance(user_ans, f"{question_context} {ans}")
            else:
                score = relevance(user_ans, ans)
            relevance_scores.append(score)
        
        max_relevance = max(relevance_scores) if relevance_scores else 0
        avg_relevance = sum(relevance_scores)/len(relevance_scores) if relevance_scores else 0

        # Creativity scoring logic
        if max_relevance >= 9.5:  # Exact or near-exact match
            return 1  # Not creative (perfect match)
            
        elif max_relevance >= 7.0:  # Close match
            return 3  # Slightly creative
            
        elif max_relevance >= 4.0:  # Somewhat related
            # More creative if partially relevant
            creativity = 5 + (1 - (max_relevance/10)) * 5
            return min(8, max(4, round(creativity)))
            
        else:  # Not relevant
            # Check for interesting patterns in wrong answers
            creativity = 6
            # Bonus for longer answers
            if len(user_ans.split()) > 2:
                creativity += 1
            # Bonus for varied vocabulary
            unique_words = len(set(user_ans.split()))
            if unique_words > 3:
                creativity += 1
            # Penalty for very short answers
            if len(user_ans) < 4:
                creativity -= 2
            return min(10, max(3, creativity))
            
    except Exception as e:
        print(f"Creativity scoring error: {e}")
        return 5  # Neutral score on error

random_words = [
    "pizza", "guitar", "mountain", "banana", "o", "chair", "coffee", "dog", 
    "umbrella", "bicycle", "lamp", "kite", "mirror", "piano", "rocket", "candle", 
    "butterfly", "hammer", "telescope", "train", "apple", "car", "book", "phone", 
    "desk", "window", "cloud", "river", "flower", "shoe", "clock", "garden", 
    "camera", "bridge", "stove", "keyboard", "moon", "star", "river", "sand",
    "glove","theme","gala","gazebo","painting","exam","mouse","computer","clip",
    "handle","laugh","cry","wifi","bell","coma","teddy bear","box","arrow","bow"
]
def generate_distractors(num_words=10):
    return ','.join(random.sample(random_words, num_words))

def topic_adherence(transcription, expected_topic, chunk_size=30):
    if not transcription.strip():
        return 1.0  # Minimum score for empty input

    # Improved sentence splitting
    sentences = [s.strip() for s in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', transcription) if s.strip()]
    
    # Create chunks (~100 words ≈ 30 seconds)
    chunks = []
    current_chunk = []
    word_count = 0
    
    for sentence in sentences:
        words = sentence.split()
        word_count += len(words)
        current_chunk.append(sentence)
        
        if word_count >= 100:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            word_count = 0
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    if not chunks:
        return 1.0

    try:
        # FIX 1: Ensure we pass lists to encode()
        topic_embedding = embedding_model.encode([expected_topic])[0]  # Get first (and only) embedding
        chunk_embeddings = embedding_model.encode(chunks)  # Already a list
        
        # FIX 2: Reshape for cosine_similarity
        topic_embedding_2d = topic_embedding.reshape(1, -1)
        chunk_embeddings_2d = chunk_embeddings.reshape(len(chunks), -1)
        
        similarities = cosine_similarity(topic_embedding_2d, chunk_embeddings_2d)[0]
        
        # Scoring logic
        avg_similarity = np.mean(similarities)
        on_topic_ratio = np.mean([s > 0.3 for s in similarities])
        
        raw_score = (avg_similarity + 1) * 2.5
        coverage_boost = 1 + (on_topic_ratio * 2)
        final_score = raw_score * coverage_boost
        
        return min(max(round(final_score, 2), 1), 10)
    
    except Exception as e:
        print(f"Error in topic_adherence: {e}")
        return 1.0  # Fallback score
    
def speech_coherence(transcription):
    if not transcription.strip():
        return 0.0  # No speech recorded

    # Split into sentences
    sentences = re.split(r'[.!?]', transcription)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Calculate readability score (normalized to 0-1)
    readability_score = textstat.flesch_reading_ease(transcription) / 100

    # Calculate semantic similarity between sentences using spaCy
    similarity_scores = []
    for i in range(len(sentences) - 1):
        doc1 = nlp(sentences[i])
        doc2 = nlp(sentences[i + 1])
        similarity_scores.append(doc1.similarity(doc2))

    avg_similarity = sum(similarity_scores) / max(len(similarity_scores), 1) if similarity_scores else 0

    # Calculate lexical diversity (unique words / total words)
    words = re.findall(r'\b\w+\b', transcription.lower())
    unique_words = set(words)
    lexical_diversity = len(unique_words) / max(len(words), 1)

    # Penalty for filler words (less severe)
    filler_words = ["um", "uh", "like", "you know"]
    filler_count = sum(transcription.lower().count(word) for word in filler_words)
    filler_penalty = min(filler_count / 20, 0.5)  # Max penalty of 0.5 for 20+ filler words

    # Final score (weighted)
    final_score = (
        (readability_score * 0.3) +  # Weight for readability
        (avg_similarity * 0.3) +     # Weight for sentence similarity
        (lexical_diversity * 0.3) -  # Weight for lexical diversity
        (filler_penalty * 0.1)       # Weight for filler words
    )

    # Normalize to a score between 1 and 10
    normalized_score = final_score * 10
    return round(max(min(normalized_score, 10), 1), 2)  # Ensure score is between 1 and 10

def speech_continuity(audio_file):
    # First check if file exists and is not empty
    if not os.path.exists(audio_file) or os.path.getsize(audio_file) == 0:
        return 0  # Invalid file
    
    try:
        # Load the audio file
        audio = AudioSegment.from_wav(audio_file)
        
        # Check if audio is silent
        if audio.dBFS < -50:  # Very quiet audio
            return 0  # Essentially silent
            
        # Detect silent intervals
        silent_intervals = detect_silence(audio, min_silence_len=500, silence_thresh=-40)
        
        # If no silence is detected, return a perfect score
        if not silent_intervals:
            return 10  # No silence detected, perfect score
        
        # Calculate the end time of the first silence in seconds
        first_silence = silent_intervals[0][1] / 1000  # Convert milliseconds to seconds

        # Score based on the timing of the first silence
        if first_silence < 1:
            return 10
        elif first_silence < 2:
            return 7
        elif first_silence < 3:
            return 4
        else:
            return 2
    except Exception as e:
        print(f"Error processing audio for continuity: {e}")
        return 0  # Error occurred

def analyze_voice(file_path, expected_mood):
    # Load the audio file
    y, sr = librosa.load(file_path)

    # Extract audio features
    pitch = np.mean(librosa.yin(y, fmin=50, fmax=300))  # Extract average pitch
    energy = np.mean(librosa.feature.rms(y=y))          # Extract average energy (volume)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)      # Extract tempo

    # Mood-based evaluation ranges
    mood_scores = {
        "Excited": {"pitch": (150, 300), "energy": (0.01, 0.05), "tempo": (80, 150)},
        "Calm": {"pitch": (80, 150), "energy": (0.005, 0.02), "tempo": (40, 80)},
        "Angry": {"pitch": (180, 300), "energy": (0.02, 0.08), "tempo": (100, 200)},
        "Sad": {"pitch": (50, 100), "energy": (0.002, 0.01), "tempo": (30, 70)},
        "Confident": {"pitch": (120, 250), "energy": (0.015, 0.05), "tempo": (60, 120)},
        "Encouraging": {"pitch": (130, 220), "energy": (0.01, 0.04), "tempo": (70, 130)},
        "Playful": {"pitch": (140, 260), "energy": (0.02, 0.06), "tempo": (90, 160)},
        "Nervous": {"pitch": (100, 180), "energy": (0.005, 0.03), "tempo": (50, 100)},
        "Determined": {"pitch": (150, 250), "energy": (0.02, 0.07), "tempo": (80, 140)},
        "Hopeful": {"pitch": (110, 200), "energy": (0.01, 0.03), "tempo": (60, 110)}
    }

    # Get expected ranges for the mood
    expected_ranges = mood_scores.get(expected_mood)
    if not expected_ranges:
        raise ValueError(f"Expected mood '{expected_mood}' not found in mood_scores.")

    # Generate Scores (scaled from 1-10)
    def calculate_score(value, min_range, max_range):
        if min_range >= max_range:
            return 1  # Avoid division by zero or invalid ranges
        if value < min_range:
            return 1  # Minimum score if below range
        if value > max_range:
            return 10  # Maximum score if above range
        return int(((value - min_range) / (max_range - min_range)) * 10)

    # Calculate scores for each feature
    pitch_score = calculate_score(pitch, expected_ranges["pitch"][0], expected_ranges["pitch"][1])
    volume_score = calculate_score(energy, expected_ranges["energy"][0], expected_ranges["energy"][1])
    pace_score = calculate_score(tempo, expected_ranges["tempo"][0], expected_ranges["tempo"][1])
    expression_score = int((pitch_score + volume_score + pace_score) / 3)  # Average of all for expression

    # Generate feedback based on feature scores
    def generate_feedback(feature, value, min_range, max_range):
        if value < min_range:
            return f"Your {feature} is too low. Try increasing it to match the mood."
        elif value > max_range:
            return f"Your {feature} is too high. Try decreasing it to match the mood."
        else:
            return f"Your {feature} is well-suited for the mood."

    # Generate feedback for each feature
    pitch_feedback = generate_feedback("pitch", pitch, expected_ranges["pitch"][0], expected_ranges["pitch"][1])
    volume_feedback = generate_feedback("volume", energy, expected_ranges["energy"][0], expected_ranges["energy"][1])
    pace_feedback = generate_feedback("pace", tempo, expected_ranges["tempo"][0], expected_ranges["tempo"][1])

    # Generate feedback for overall expression
    if expression_score < 4:
        expression_feedback = "Your overall expression needs improvement. Try adjusting pitch, volume, and pace."
    elif expression_score < 7:
        expression_feedback = "Your overall expression is decent but could be improved."
    else:
        expression_feedback = "Your overall expression is balanced and matches the mood well."

    # Return results
    return {
        "mood": expected_mood,
        "pitch_score": pitch_score,
        "pitch_feedback": pitch_feedback,
        "volume_score": volume_score,
        "volume_feedback": volume_feedback,
        "pace_score": pace_score,
        "pace_feedback": pace_feedback,
        "expression_score": expression_score,
        "expression_feedback": expression_feedback
    }





