from flask import Flask, render_template, redirect, url_for,session,request,flash,jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import JSON
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from sqlalchemy.orm import Session
import librosa
import numpy as np
from sentence_transformers import SentenceTransformer, util
from pydub import AudioSegment, silence
import random
import time
import os
from fuzzywuzzy import fuzz
from utils import convert_to_wav,is_valid_wav
from utils import transcribe_audio_base,transcribe_audio_small,creativity,relevance,timing,speech_continuity,is_valid_wav
from utils import generate_distractors,topic_adherence,speech_coherence
from utils import analyze_voice
import whisper

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
load_dotenv() 
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
# Path to the FFmpeg executable
ffmpeg_path = r"F:/ffmpeg-7.1-full_build/ffmpeg-7.1-full_build/bin"
os.environ["PATH"] += os.pathsep +ffmpeg_path

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class Users(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(250), unique=True,nullable=False)
    password = db.Column(db.String(250),nullable=False)
    exercise1_scores = db.relationship('exercise1_Score', backref='user', lazy=True)
    exercise2_scores = db.relationship('exercise2_Score', backref='user', lazy=True)
    exercise3_scores = db.relationship('exercise3_score', backref='user', lazy=True)

class Questions_1(db.Model):
    q_id = db.Column(db.Integer, primary_key=True)  # Unique ID for the question
    ques = db.Column(db.String(350), nullable=False)  # The question text
    ans = db.Column(JSON, nullable=False)  # Store multiple correct answers as a JSON array
    context = db.Column(db.String(350), nullable=True)  # Context of the question (e.g., "quick like _____")
    exercise_id = db.Column(db.Integer, nullable=False)  # ID of the exercise

#store scores of rapid fire in db
class exercise1_Score(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timing_score = db.Column(db.Integer, nullable=False)
    relevance_score=db.Column(db.Integer, nullable=False)
    creativity_score=db.Column(db.Integer, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

#feedback of rapid fire
class exercise1_feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(50), nullable=False)  #Timing, Creativity, Relevance
    score_range = db.Column(db.String(20), nullable=False)  # "high" or "low"
    message = db.Column(db.Text, nullable=False)  # Feedback message

#table to store questions of triple_step
class Questions_2(db.Model):
    q_id = db.Column(db.Integer, primary_key=True)  
    topic = db.Column(db.String(200), nullable=False)  
    distractors = db.Column(db.Text, nullable=False)  
    exercise_id = db.Column(db.Integer, nullable=False) 

#store scores of  triple_step in db
class exercise2_Score(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    adherence_score = db.Column(db.Integer, nullable=False)
    coherence_score=db.Column(db.Integer, nullable=False)
    continuity_score=db.Column(db.Integer, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

#store feedback of triple_step
class exercise2_feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(50), nullable=False)  # Topic Adherence, Speech Coherence,continuity
    score_range = db.Column(db.String(20), nullable=False)  # "high" or "low"
    message = db.Column(db.Text, nullable=False)  # Feedback message

#table to store questions of conductor
class Questions_3(db.Model):
    q_id = db.Column(db.Integer, primary_key=True)  
    mood = db.Column(db.String(50), nullable=False)
    sentence=db.Column(db.String(50), nullable=False)
    exercise_id = db.Column(db.Integer, nullable=False) 

#store scores of conductor in db
class exercise3_score(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    mood = db.Column(db.String(50), nullable=False)  
    energy_level = db.Column(db.Float, nullable=False)  
    pitch_variation = db.Column(db.Float, nullable=False) 
    pacing = db.Column(db.Float, nullable=False)  
    emotional_clarity = db.Column(db.Float, nullable=False)  

#store feedback for conductor
class exercise3_feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(50), nullable=False)  # e.g., Pitch, Volume, Pace, Expression
    score_range = db.Column(db.String(20), nullable=False)  # "high" or "low"
    message = db.Column(db.Text, nullable=False)  # Feedback message

#overall_feedback
class final_feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(50), nullable=False)  # e.g., Overall Performance
    score_range = db.Column(db.String(20), nullable=False)  # "high", "medium", or "low"
    message = db.Column(db.Text, nullable=False)  # Feedback message


with app.app_context():
    db.drop_all()
    db.create_all() # Ensure tables exist

    # Check if questions already exist to prevent duplicates
    if Questions_1.query.count() == 0:
        exercise1 = [
            Questions_1(
                q_id=1,
                ques="Time is _______",
                ans="money",
                context="Time is _______",
                exercise_id=1
            ),
            Questions_1(
                q_id=2,
                ques="Patience is like a _______",
                ans="seed",
                context="Patience is like a _______",
                exercise_id=1
            ),
            Questions_1(
                q_id=3,
                ques="Wisdom is like a _______",
                ans="tree",
                context="Wisdom is like a _______",
                exercise_id=1
            ),
            Questions_1(
                q_id=4,
                ques="Hope is like a _______",
                ans="lighthouse",
                context="Hope is like a _______",
                exercise_id=1
            ),
            Questions_1(
                q_id=5,
                ques="Busy like a _______",
                ans="bee",
                context="Busy like a _______",
                exercise_id=1
            ),
            Questions_1(
                q_id=6,
                ques="Curiosity is like _______",
                ans="cat",
                context="Curiosity is like _______",
                exercise_id=1
            ),
            Questions_1(
                q_id=7,
                ques="Quick like a _______",
                ans="cheetah",
                context="Quick like a _______",
                exercise_id=1
            ),
            Questions_1(
                q_id=8,
                ques="Knowledge is like _______",
                ans="river",
                context="Knowledge is like _______",
                exercise_id=1
            ),
            Questions_1(
                q_id=9,
                ques="Trust is like _______",
                ans="glass",
                context="Trust is like _______",
                exercise_id=1
            ),
            Questions_1(
                q_id=10,
                ques="Success is like _______",
                ans="ladder",
                context="Success is like _______",
                exercise_id=1
            ),
        ]
        db.session.add_all(exercise1)
        db.session.commit()

    if Questions_2.query.count() == 0:
        exercise1 =[
            Questions_2(q_id=1,topic="Global Warming",distractors=generate_distractors(10),exercise_id=2),
            Questions_2(q_id=2,topic="Climate Change", distractors=generate_distractors(10),exercise_id=2),
            Questions_2(q_id=3,topic="The impact of technology on society",distractors=generate_distractors(10),exercise_id=2),
            Questions_2(q_id=4,topic="The importance of maintaining a healthy diet",distractors=generate_distractors(10),exercise_id=2),
            Questions_2(q_id=5,topic="Artificial Intelligence", distractors=generate_distractors(10),exercise_id=2)
        ]
        db.session.add_all(exercise1)
        db.session.commit()

    if Questions_3.query.count() == 0:
        exercise3 = [
            Questions_3(q_id=1, mood="Excited", sentence="I got the Job!", exercise_id=3),
            Questions_3(q_id=2, mood="Calm", sentence="Let's calm down and take a deep breath.", exercise_id=3),
            Questions_3(q_id=3, mood="Angry", sentence="I can't believe I let this happen!", exercise_id=3),
            Questions_3(q_id=4, mood="Sad", sentence="I wish i could go back home.", exercise_id=3),
            Questions_3(q_id=5, mood="Confident", sentence="I know I can achieve anything I set my mind to.", exercise_id=3),
            Questions_3(q_id=6, mood="Encouraging", sentence="Keep pushing forward, you've got this!", exercise_id=3),
            Questions_3(q_id=7, mood="Playful", sentence="Let's make this an adventure!", exercise_id=3),
            Questions_3(q_id=8, mood="Nervous", sentence="I hope this goes well...", exercise_id=3),
            Questions_3(q_id=9, mood="Determined", sentence="Nothing can stop me from reaching my goal.", exercise_id=3),
            Questions_3(q_id=10, mood="Hopeful", sentence="I hope i got selected.", exercise_id=3)
        ]

        db.session.add_all(exercise3)
        db.session.commit()

    if exercise1_feedback.query.count() == 0:
        feedback_entries = [
            # Timing feedback
            exercise1_feedback(category="Timing", score_range="high", message="Great job! You responded quickly and confidently. Keep up the good pace!"),
            exercise1_feedback(category="Timing", score_range="medium", message="Your response time was decent, but try to respond a bit faster next time."),
            exercise1_feedback(category="Timing", score_range="low", message="Try to respond a bit faster next time."),
            
            # Creativity feedback
            exercise1_feedback(category="Creativity", score_range="high", message="Wow, that was a creative answer! You're thinking outside the box!"),
            exercise1_feedback(category="Creativity", score_range="medium", message="Your answer was creative, but you can push it further with more originality."),
            exercise1_feedback(category="Creativity", score_range="low", message="Your answer was good, but try to add more originality next time!"),
            
            # Relevance feedback
            exercise1_feedback(category="Relevance", score_range="high", message="Perfect! Your answer was spot on and highly relevant."),
            exercise1_feedback(category="Relevance", score_range="medium", message="Your answer was relevant, but try to focus more on the topic next time."),
            exercise1_feedback(category="Relevance", score_range="low", message="Your answer was close, but try to focus more on the topic next time."),
            
        ]
        db.session.add_all(feedback_entries)
        db.session.commit()

    if exercise2_feedback.query.count() == 0:
        feedback_entries = [
            # Topic Adherence feedback
            exercise2_feedback(category="Topic Adherence", score_range="high", message="You stayed perfectly on topic. Well done!"),
            exercise2_feedback(category="Topic Adherence", score_range="medium", message="You mostly stayed on topic, but try to focus more on the main idea."),
            exercise2_feedback(category="Topic Adherence", score_range="low", message="You strayed a bit from the topic. Try to focus more on the main idea."),
            
            # Speech Coherence feedback
            exercise2_feedback(category="Speech Coherence", score_range="high", message="Your speech was clear and well-structured. Great job!"),
            exercise2_feedback(category="Speech Coherence", score_range="medium", message="Your speech was clear, but it could be more organized."),
            exercise2_feedback(category="Speech Coherence", score_range="low", message="Your speech could be more organized. Try to structure your thoughts better."),

            # Continuity feedback
            exercise2_feedback(category="Continuity", score_range="high", message="Your speech was smooth and uninterrupted. Great job!"),
            exercise2_feedback(category="Continuity", score_range="medium", message="Your speech was mostly smooth, but try to avoid pauses or filler words."),
            exercise2_feedback(category="Continuity", score_range="low", message="Try to avoid pauses or filler words. Practice speaking more fluently!"),
        ]
        db.session.add_all(feedback_entries)
        db.session.commit()

    if exercise3_feedback.query.count() == 0:
        feedback_entries = [
            # Pitch feedback
            exercise3_feedback(category="Pitch", score_range="high", message="Your pitch was varied and engaging! Well done!"),
            exercise3_feedback(category="Pitch", score_range="medium", message="Your pitch was good, but try adding more variation to make your speech more expressive."),
            exercise3_feedback(category="Pitch", score_range="low", message="Try adding more pitch variation to make your speech more expressive."),
            
            # Volume feedback
            exercise3_feedback(category="Volume", score_range="high", message="Great volume control! Your speech was clear and well-balanced."),
            exercise3_feedback(category="Volume", score_range="medium", message="Your volume was good, but try adjusting it to match the mood and emphasize key points."),
            exercise3_feedback(category="Volume", score_range="low", message="Try adjusting your volume to match the mood and emphasize key points."),
            
            # Pace feedback
            exercise3_feedback(category="Pace", score_range="high", message="Your pace was perfect! Well done."),
            exercise3_feedback(category="Pace", score_range="medium", message="Your pace was good, but try to control your speed better to ensure clarity."),
            exercise3_feedback(category="Pace", score_range="low", message="Try to control your speed better to ensure clarity."),
            
            # Expression feedback
            exercise3_feedback(category="Expression", score_range="high", message="Your voice had great emotional expression!"),
            exercise3_feedback(category="Expression", score_range="medium", message="Your expression was good, but try to match your tone more closely to the emotions you're conveying."),
            exercise3_feedback(category="Expression", score_range="low", message="Try to match your tone more closely to the emotions you're conveying.")
        ]
        db.session.add_all(feedback_entries)
        db.session.commit()

    if final_feedback.query.count() == 0:
        feedback_entries = [
            final_feedback(category="Overall Performance", score_range="high", message="Excellent performance! Your speech was engaging and well-structured."),
            final_feedback(category="Overall Performance", score_range="medium", message="Good effort! Your speech was clear, but there's room for improvement in expression and structure."),
            final_feedback(category="Overall Performance", score_range="low", message="Keep practicing! Focus on clarity, confidence, and expression."),
        ]
        db.session.add_all(feedback_entries)
        db.session.commit()

@login_manager.user_loader
def loader_user(user_id):
    return db.session.get(Users, int(user_id))

@app.route("/")
def home():
    return render_template("index.html")  

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = Users.query.filter_by(username=username).first()
        if not user:
            #if not get_flashed_messages():
                flash("User does not exist. Please register first.", "danger")
                return redirect(url_for("login"))

        if not check_password_hash(user.password, password):
            flash("Incorrect password. Try again.", "danger")
            return redirect(url_for("login"))

        login_user(user) #log the user in
        flash("Login successful!", "success")
        return redirect(url_for("rapid_fire_instructions"))
    return render_template("login.html")  

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the user already exists
        user = Users.query.filter_by(username=username).first()
        if user:
            flash('User already exists. Please choose a different username.', 'error')
            return redirect(url_for('register'))

        # Hash the password
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        # Create a new user
        new_user = Users(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')  


@app.route("/rapid_fire_instructions",methods=["GET", "POST"])
@login_required 
def rapid_fire_instructions():
     if request.method=="POST":
          return redirect(url_for("rapid_fire"))
     return render_template("rapid_fire_instructions.html")

@app.route("/rapid_fire", methods=["GET", "POST"])
@login_required
def rapid_fire():
    user_id = current_user.id
    exercise1_questions = Questions_1.query.filter_by(exercise_id=1).all()
    #correct_ans = {q.q_id: q.ans for q in exercise1_questions}  # Dictionary of correct answers
    scores = []

    if request.method == "POST":
        for question in exercise1_questions:
            audio_file = request.files.get(f"audio_{question.q_id}")

            if audio_file:
                # Create unique file name for each user-question
                webm_path = os.path.join("uploads", f"user_{user_id}_q{question.q_id}.webm")
                wav_path = webm_path.replace(".webm", ".wav")

                # Ensure the uploads directory exists
                os.makedirs(os.path.dirname(webm_path), exist_ok=True)

                # Save the uploaded file
                audio_file.save(webm_path)
                print(f"Saved audio for question {question.q_id} at {webm_path}")  # Debugging

                # Convert .webm to .wav
                convert_to_wav(webm_path, wav_path)
                print(f"Converted {webm_path} to {wav_path}")  # Debugging

                # Validate WAV file
                if is_valid_wav(wav_path):
                    # Transcribe audio
                    user_ans = transcribe_audio_base(wav_path)
                    print(f"Question {question.q_id}: Transcription = {user_ans}")

                    if user_ans:  # This now properly checks if transcription returned something valid
                        # Calculate scores
                        # Get the recording duration directly from the form instead of calculating it
                        try:
                            # Use recording_duration directly instead of calculating from timestamps
                            recording_duration = float(request.form.get(f"recording_duration_{question.q_id}", 0))
                            
                            # Ensure it's a positive value
                            if recording_duration < 0:
                                recording_duration = 0
                                
                            print(f"Question {question.q_id}: Total Time: {recording_duration}")
                            timing_score = timing(recording_duration)
                        except (ValueError, TypeError) as e:
                            print(f"Error processing timing for Question {question.q_id}: {e}")
                            timing_score = 0

                        relevance_score = relevance(user_ans, question.ans)
                        creativity_score=creativity(user_ans,question.ans)
                    else:
                        # No valid transcription
                        print(f"No valid transcription for Question {question.q_id}")
                        timing_score = 0
                        relevance_score = 0
                        creativity_score=0
                else:
                    # Invalid WAV file
                    print(f"Invalid WAV file for Question {question.q_id}")
                    timing_score = 0
                    relevance_score = 0
                    creativity_score=0

                scores.append({
                    "q_id": question.q_id,
                    "timing": timing_score,
                    "relevance": relevance_score,
                    "creativity":creativity_score
                })

                # Store the score in the database
                new_score = exercise1_Score(
                    user_id=user_id,
                    timing_score=timing_score,
                    relevance_score=relevance_score,
                    creativity_score=creativity_score
                )
                db.session.add(new_score)

        db.session.commit()
        session["rapid_fire_scores"] = scores
        session["rapid_fire_questions"] = [{"q_id": q.q_id, "ques": q.ques} for q in exercise1_questions]

        flash("Scores have been recorded!", "success")
        return redirect(url_for("rapid_fire_feedback"))

    return render_template("rapid_fire.html", questions=exercise1_questions, time=time)


def get_feedback_rf(category, score, feedback_model):
    if score > 8:  
        feedback = feedback_model.query.filter_by(category=category, score_range="high").first()
    elif  6<= score <= 8:  
        feedback = feedback_model.query.filter_by(category=category, score_range="medium").first()
    else:  
        feedback = feedback_model.query.filter_by(category=category, score_range="low").first()
    
    return feedback.message if feedback else "No feedback available."

@app.route("/rapid_fire_feedback", methods=["GET", "POST"])
@login_required
def rapid_fire_feedback():
    scores = session.get("rapid_fire_scores", [])
    questions = session.get("rapid_fire_questions", [])
    feedback = []

    for score in scores:
        fb = {
            "timing_feedback": get_feedback_rf("Timing", score["timing"], exercise1_feedback),
            "relevance_feedback": get_feedback_rf("Relevance", score["relevance"], exercise1_feedback),
            "creativity_feedback": get_feedback_rf("Creativity", score["creativity"], exercise1_feedback)
        }
        feedback.append(fb)
    

    if request.method == "POST":
        return redirect(url_for("triple_step_instructions"))
    return render_template("rapid_fire_feedback.html", scores=scores, feedback=feedback, questions=questions, zip=zip)


@app.route("/triple_step_instructions",methods=["GET", "POST"])
@login_required 
def triple_step_instructions():
     if request.method=="POST":
          return redirect(url_for("triple_step"))
     return render_template("triple_step_instructions.html")


@app.route("/triple_step", methods=["GET", "POST"])
@login_required
def triple_step():
    user_id = current_user.id
    
    if request.method == "GET":
        # Generate and store topic ID in session
        topic_id = random.randint(1, 5)
        session['current_topic_id'] = topic_id
        question = Questions_2.query.get(topic_id)
        if not question:
            flash("Topic not found", "error")
            return redirect(url_for("triple_step"))
            
        return render_template("triple_step.html", 
                            topic=question.topic,
                            distractors=question.distractors.split(','))

    elif request.method == "POST":
        # Retrieve the original topic from session
        topic_id = session.get('current_topic_id')
        if not topic_id:
            flash("Session expired. Please try again.", "error")
            return redirect(url_for("triple_step"))
            
        question = Questions_2.query.get(topic_id)
        if not question:
            flash("Topic not found", "error")
            return redirect(url_for("triple_step"))

        # Audio processing
        audio_file = request.files.get("audio_file")
        if not audio_file:
            flash("No audio file uploaded", "error")
            return redirect(url_for("triple_step"))

        # Secure file handling
        try:
            webm_path = f"uploads/user_{user_id}_topic_{topic_id}_{int(time.time())}.webm"
            wav_path = webm_path.replace(".webm", ".wav")
            os.makedirs(os.path.dirname(webm_path), exist_ok=True)
            audio_file.save(webm_path)
            convert_to_wav(webm_path, wav_path)

            # Debug logging (terminal only)
            print(f"\n🔍 [DEBUG] Processing audio for user {user_id}")
            print(f"Topic: {question.topic}")
            
            # Transcription
            user_transcription = transcribe_audio_small(wav_path)
            print(f"Raw transcription: {user_transcription}")

            # Scoring
            topic_adherence_score = topic_adherence(user_transcription, question.topic)
            speech_coherence_score = speech_coherence(user_transcription)
            speech_continuity_score = speech_continuity(wav_path)
            print(f"Scores - Adherence: {topic_adherence_score}, Coherence: {speech_coherence_score}, Continuity: {speech_continuity_score}")

            # Store results
            new_score = exercise2_Score(
                user_id=user_id,
                adherence_score=topic_adherence_score,
                coherence_score=speech_coherence_score,
                continuity_score=speech_continuity_score
            )
            db.session.add(new_score)
            db.session.commit()

            # Prepare feedback (without exposing transcription)
            session["triple_step_scores"] = {
                "adherence_score": topic_adherence_score,
                "coherence_score": speech_coherence_score,
                "continuity_score": speech_continuity_score,
                "topic": question.topic  # For reference but not displayed
            }

            flash("Your speech has been evaluated!", "success")
            return redirect(url_for("triple_step_feedback"))

        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            flash("Error processing your audio. Please try again.", "error")
            return redirect(url_for("triple_step"))

def get_feedback_ts(category, score, topic_adherence_score, feedback_model):
    # Define thresholds
    HIGH_THRESHOLD = 7
    MEDIUM_THRESHOLD = 4

    # Special case: If topic adherence is too low, other feedback should also be harsher
    if topic_adherence_score < MEDIUM_THRESHOLD and category != "Topic Adherence":
        score_range = "low"  # Override for Speech Coherence & Continuity
    else:
        if score > HIGH_THRESHOLD:
            score_range = "high"
        elif score > MEDIUM_THRESHOLD:
            score_range = "medium"
        else:
            score_range = "low"

    # Fetch feedback from database
    feedback = feedback_model.query.filter_by(category=category, score_range=score_range).first()
    return feedback.message if feedback else "No feedback available."


@app.route("/triple_step_feedback", methods=["GET", "POST"])
@login_required
def triple_step_feedback():
    # Retrieve scores from the session
    scores = session.get("triple_step_scores", {})
    
    adherence_score = scores.get("adherence_score", 0)
    coherence_score = scores.get("coherence_score", 0)
    continuity_score = scores.get("continuity_score", 0)

    # Retrieve feedback messages based on scores
    adherence_feedback = get_feedback_ts("Topic Adherence", adherence_score, adherence_score, exercise2_feedback)
    coherence_feedback = get_feedback_ts("Speech Coherence", coherence_score, adherence_score, exercise2_feedback)
    continuity_feedback = get_feedback_ts("Continuity", continuity_score, adherence_score, exercise2_feedback)

    # Handle POST request (e.g., when the user clicks "Next")
    if request.method == "POST":
        return redirect(url_for("conductor_instructions"))

    # Render the feedback template with scores and feedback
    return render_template("triple_step_feedback.html", 
                           adherence_score=adherence_score,
                           coherence_score=coherence_score,
                           continuity_score=continuity_score, 
                           adherence_feedback=adherence_feedback,
                           coherence_feedback=coherence_feedback,
                           continuity_feedback=continuity_feedback)


@app.route("/conductor_instructions", methods=["GET", "POST"])
@login_required 
def conductor_instructions():
    if request.method == "POST":
        return redirect(url_for("conductor"))
    return render_template("conductor_instructions.html")


@app.route("/conductor", methods=["GET", "POST"])
@login_required
def conductor():
    moods = Questions_3.query.all()

    if request.method == "POST":
        user_id = current_user.id
        mood_qid = request.form.get("mood_qid")
        audio_file = request.files.get("audio_file")

        # Validate mood selection
        mood_entry = Questions_3.query.get(mood_qid)
        if not audio_file or not mood_entry:
            flash("Invalid input. Please try again.", "danger")
            return redirect(url_for("conductor"))

        # Save audio file as .webm
        webm_path = f"uploads/user_{user_id}_mood_{mood_entry.mood}.webm"
        wav_path = webm_path.replace(".webm", ".wav")

        # Ensure the uploads directory exists
        os.makedirs(os.path.dirname(webm_path), exist_ok=True)

        # Save the uploaded file
        audio_file.save(webm_path)
        print(f"Saved audio for mood {mood_entry.mood} at {webm_path}")  # Debugging

        # Convert .webm to .wav
        convert_to_wav(webm_path, wav_path)
        print(f"Converted {webm_path} to {wav_path}")  # Debugging

        # Analyze the voice
        feedback = analyze_voice(wav_path, mood_entry.mood)

        # Store feedback in session
        session["conductor_feedback"] = feedback
        session["conductor_scores"] = {  
            "pitch_score": feedback.get("pitch_score", 0),
            "volume_score": feedback.get("volume_score", 0),
            "pace_score": feedback.get("pace_score", 0),
            "expression_score": feedback.get("expression_score", 0),
        }
        session.modified = True 
        session.modified = True  # Ensure the session is saved

        # Redirect to the conductor_feedback route
        return redirect(url_for("conductor_feedback"))

    return render_template("conductor.html", moods=moods)

def get_feedback_c(category, score, feedback_model):
    
    HIGH_THRESHOLD = 7
    MEDIUM_THRESHOLD = 4

    if score > HIGH_THRESHOLD:
        score_range = "high"
    elif score > MEDIUM_THRESHOLD:
        score_range = "medium"
    else:
        score_range = "low"

    # Fetch feedback from database
    feedback = feedback_model.query.filter_by(category=category, score_range=score_range).first()
    return feedback.message if feedback else "No feedback available."

@app.route("/conductor_feedback", methods=["GET", "POST"])
@login_required
def conductor_feedback():
    try:
        print("Conductor Feedback Route Called")  # Debugging

        # Retrieve feedback from the session
        feedback = session.get("conductor_feedback", {})

        if request.method == "POST":
            return redirect(url_for("overall_feedback"))
        
        pitch_score = feedback.get("pitch_score", 0)
        volume_score = feedback.get("volume_score", 0)
        pace_score = feedback.get("pace_score", 0)
        expression_score = feedback.get("expression_score", 0)

        # Fetch feedback from the database
        pitch_feedback = get_feedback_c("Pitch", pitch_score, exercise3_feedback)
        volume_feedback = get_feedback_c("Volume", volume_score, exercise3_feedback)
        pace_feedback = get_feedback_c("Pace", pace_score, exercise3_feedback)
        expression_feedback = get_feedback_c("Expression", expression_score, exercise3_feedback)

        # Store the updated feedback in session
        session["conductor_feedback"] = {
            "pitch_score": pitch_score,
            "pitch_feedback": pitch_feedback,
            "volume_score": volume_score,
            "volume_feedback": volume_feedback,
            "pace_score": pace_score,
            "pace_feedback": pace_feedback,
            "expression_score": expression_score,
            "expression_feedback": expression_feedback
        }


        return render_template(
            "conductor_feedback.html",
            pitch_score=pitch_score,
            pitch_feedback=pitch_feedback,
            volume_score=volume_score,
            volume_feedback=volume_feedback,
            pace_score=pace_score,
            pace_feedback=pace_feedback,
            expression_score=expression_score,
            expression_feedback=expression_feedback
        )
    except Exception as e:
        print("Error in conductor_feedback route:", str(e))  # Debugging
        return "An error occurred. Check the console for details.", 500

@app.route("/overall_feedback")
@login_required
def overall_feedback():
    # Retrieve scores from the session, default to an empty list if not found
    rapid_fire_scores = session.get("rapid_fire_scores", [])
    triple_step_scores = session.get("triple_step_scores", [])
    conductor_scores = session.get("conductor_scores", [])

    def calculate_avg(scores, key):
        valid_scores = []
        for score in scores:
            # Check if score is a dictionary and contains the key
            if isinstance(score, dict) and key in score:
                valid_scores.append(score[key])
        
        # Calculate average only if we have valid scores
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0

    # Compute average scores
    avg_timing = calculate_avg(rapid_fire_scores, "timing")
    avg_adherence = calculate_avg(triple_step_scores, "adherence_score")
    avg_expression = calculate_avg(conductor_scores, "expression_score")

    # Overall performance score as the mean of the individual components
    overall_score = (avg_timing + avg_adherence + avg_expression) / 4

    # Determine the overall feedback category
    score_category = "high" if overall_score > 3 else "medium" if overall_score > 2 else "low"

    # Retrieve overall feedback from the database
    overall_fb = final_feedback.query.filter_by(category="Overall Performance", score_range=score_category).first()

    # Render the template with only the overall feedback
    return render_template(
        "overall_feedback.html",
        overall_feedback=overall_fb.message if overall_fb else "No overall feedback available."
    )


if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
