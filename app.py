from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import joblib
import pandas as pd
import os
import requests
import re
from ml.resume_parser import extract_skills, extract_resume_text, match_resume
from ml.skill_gap import skill_gap
from generate_graph import generate_ranking_graph

app = Flask(__name__)
app.secret_key = "secret_candidate_key"
os.makedirs('uploads', exist_ok=True)

model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
try:
    model = joblib.load(model_path)
except Exception as e:
    model = None
    print(f"Failed to load model: {e}")

DB_PATH = os.path.join(os.path.dirname(__file__), 'database', 'candidates.db')

def get_db_connection():
    return sqlite3.connect(DB_PATH)

import json

def generate_quiz_questions(experience, skills_str):
    prompt = f"Generate exactly 5 multiple choice questions for a technical interview. Candidate has {experience} years experience and skills: {skills_str}. Output ONLY valid JSON containing an array of objects. Each object must have 'question' (string), 'options' (array of 4 strings), and 'answer' (the exact string from options that is correct)."
    try:
        r = requests.post('http://localhost:11434/api/generate', json={
            "model": "gemma3:4b",
            "prompt": prompt,
            "stream": False
        }, timeout=30)
        json_data = r.json()
        if 'response' in json_data:
            response_text = json_data['response'].strip()
            # Try to extract just the JSON part
            match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if match:
                return json.loads(match.group())
            else:
                return json.loads(response_text)
    except Exception as e:
        print("Ollama Error (generate mcq):", e)
    
    # Fallback questions
    return [
        {"question": "What is polymorphism?", "options": ["Many forms", "Database", "Networking", "None"], "answer": "Many forms"},
        {"question": "What is inheritance?", "options": ["Class hierarchy", "Variables", "Loop", "None"], "answer": "Class hierarchy"},
        {"question": "What is encapsulation?", "options": ["Data hiding", "Speed", "Sorting", "None"], "answer": "Data hiding"},
        {"question": "What does API stand for?", "options": ["Application Programming Interface", "Apple Pie", "Array", "None"], "answer": "Application Programming Interface"},
        {"question": "What is a primary key?", "options": ["Unique identifier", "String", "Foreign key", "None"], "answer": "Unique identifier"}
    ]

def evaluate_resume(text):
    prompt = f"Evaluate the following resume text. Provide a score out of 100 based on quality, and a brief summary. Format exactly like:\nScore: [number]\nSummary: [your summary]\nResume Text: {text[:1500]}"
    try:
        r = requests.post('http://localhost:11434/api/generate', json={
            "model": "gemma3:4b",
            "prompt": prompt,
            "stream": False
        }, timeout=30)
        json_data = r.json()
        if 'response' in json_data:
            res = json_data['response'].strip()
            score = 50
            summary = "No summary provided."
            score_match = re.search(r'Score:\s*(\d+)', res)
            if score_match:
                score = min(100, max(0, int(score_match.group(1))))
            summary_match = re.search(r'Summary:\s*(.*)', res, re.DOTALL)
            if summary_match:
                summary = summary_match.group(1).strip()
            return score, summary
    except Exception as e:
        print("Ollama Error (resume eval):", e)
    return 50, "Could not evaluate resume."

@app.route('/')
def index():
    if 'user_id' in session:
        if session.get('is_admin') == 1:
            return redirect(url_for('dashboard'))
        else:
            return redirect(url_for('user_dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        u = request.form['username']
        p = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE username=? AND password=?", (u, p)).fetchone()
        conn.close()
        
        if user:
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['is_admin'] = user[3]
            
            if user[3] == 1:
                return redirect(url_for('dashboard'))
            else:
                return redirect(url_for('user_dashboard'))
        else:
            error = "Invalid credentials!"
            
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        u = request.form['username']
        p = request.form['password']
        
        conn = get_db_connection()
        try:
            conn.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, 0)", (u, p))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            conn.close()
            error = "Username already exists!"
            
    return render_template('register.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/apply', methods=['GET', 'POST'])
def apply():
    if 'user_id' not in session or session.get('is_admin') == 1:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        name = request.form['name']
        exp = int(request.form['exp'])
        resume = request.files['resume']
        
        path = os.path.join("uploads", resume.filename)
        resume.save(path)
        
        skills_count, skills_str = extract_skills(path)
        resume_text = extract_resume_text(path)
        r_score, r_summary = evaluate_resume(resume_text)
        
        conn = get_db_connection()
        jobs = conn.execute("SELECT description, skills_required FROM jobs").fetchall()
        conn.close()
        
        max_match = 0.0
        if jobs:
            for job in jobs:
                jd = f"{job[0]} {job[1]}"
                score = match_resume(resume_text, jd)
                if score > max_match:
                    max_match = score
        
        session['candidate_name'] = name
        session['candidate_exp'] = exp
        session['candidate_skills_count'] = skills_count
        session['candidate_skills_str'] = skills_str
        session['resume_score'] = r_score
        session['resume_summary'] = r_summary
        session['resume_match'] = max_match
        
        # Pre-generate questions
        questions = generate_quiz_questions(exp, skills_str)
        session['quiz_questions'] = questions
        
        return redirect(url_for('quiz'))
        
    return render_template('apply.html')

@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    if 'candidate_name' not in session:
        return redirect(url_for('apply'))
        
    questions = session.get('quiz_questions', [])
        
    if request.method == 'POST':
        correct_count = 0
        total_q = len(questions)
        for i, q in enumerate(questions):
            user_answer = request.form.get(f'answer_{i}')
            if user_answer == q.get('answer'):
                correct_count += 1
                
        score = int((correct_count / total_q) * 100) if total_q > 0 else 0
        
        exp = session['candidate_exp']
        skills_count = session['candidate_skills_count']
        skills_str = session['candidate_skills_str']
        resume_match = session.get('resume_match', 0.0)
        
        result = 0
        if model:
            try:
                # The model expects [exp, skills, quiz] exactly
                prediction = model.predict([[exp, skills_count, score]])[0]
                result = int(prediction)
            except Exception as e:
                print("Model prediction error:", e)
        
        conn = get_db_connection()
        try:
            conn.execute("INSERT INTO candidates (user_id, name, experience, skills, quiz, selected, resume_score, summary, skills_list, resume_match) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         (session['user_id'], session['candidate_name'], exp, skills_count, score, result, session.get('resume_score', 0), session.get('resume_summary', ''), skills_str, resume_match))
        except sqlite3.OperationalError:
            # Fallback if skills_list column doesn't exist
            conn.execute("INSERT INTO candidates (user_id, name, experience, skills, quiz, selected, resume_score, summary) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (session['user_id'], session['candidate_name'], exp, skills_count, score, result, session.get('resume_score', 0), session.get('resume_summary', '')))

        conn.commit()
        conn.close()
        
        session['last_result'] = result
        session['last_score'] = score
        
        # lock them out of resubmitting quiz by clearing application cache
        session.pop('candidate_name', None)
        session.pop('quiz_questions', None)
        
        return redirect(url_for('user_dashboard'))
        
    return render_template('quiz.html', questions=questions)

@app.route('/user_dashboard')
def user_dashboard():
    if 'user_id' not in session or session.get('is_admin') == 1:
        return redirect(url_for('index'))
        
    conn = get_db_connection()
    c = conn.cursor()
    candidate = c.execute("SELECT * FROM candidates WHERE user_id=? ORDER BY id DESC LIMIT 1", (session['user_id'],)).fetchone()
    jobs = c.execute("SELECT * FROM jobs").fetchall()
    conn.close()
    
    matched_jobs = []
    if candidate:
        quiz_score = candidate[5]
        resume_score = candidate[7] or 0
        candidate_skills = candidate[9] or ""
        for job in jobs:
            if quiz_score >= job[4] and resume_score >= job[5]:
                missing = skill_gap(candidate_skills, job[3])
                matched_jobs.append(job + (missing,))
                
    return render_template('user_dashboard.html', candidate=candidate, matched_jobs=matched_jobs)

@app.route('/admin/job/add', methods=['POST'])
def add_job():
    if 'user_id' not in session or session.get('is_admin') == 0:
        return redirect(url_for('index'))
    title = request.form['title']
    desc = request.form['description']
    skills = request.form['skills_required']
    min_quiz = float(request.form.get('min_quiz_score', 10))
    min_resume = float(request.form.get('min_resume_score', 75))
    
    conn = get_db_connection()
    conn.execute("INSERT INTO jobs (title, description, skills_required, min_quiz_score, min_resume_score) VALUES (?, ?, ?, ?, ?)",
                 (title, desc, skills, min_quiz, min_resume))
    conn.commit()
    conn.close()
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session or session.get('is_admin') == 0:
        return redirect(url_for('index'))
        
    conn = get_db_connection()
    c = conn.cursor()
    data = c.execute("SELECT * FROM candidates ORDER BY id DESC").fetchall()
    jobs = c.execute("SELECT * FROM jobs ORDER BY id DESC").fetchall()
    conn.close()
    
    # Generate individual job ranking graphs
    job_graphs = []
    
    for job in jobs:
        job_id = job[0]
        graph_filename = f'ranking_job_{job_id}.png'
        graph_path = os.path.join(app.root_path, 'static', graph_filename)
        # Attempt to create dynamic specific graph, if it works, append to list
        if generate_ranking_graph(DB_PATH, graph_path, job_id=job_id):
            job_graphs.append({'id': job_id, 'title': job[1], 'filename': graph_filename})
            
    # As a fallback, render the generic one as well just in case they want a macro perspective
    global_graph_path = os.path.join(app.root_path, 'static', 'ranking.png')
    global_graph_exists = generate_ranking_graph(DB_PATH, global_graph_path)
    if global_graph_exists:
        job_graphs.insert(0, {'id': 'global', 'title': 'Overall Macro Leaderboard', 'filename': 'ranking.png'})
    
    return render_template('dashboard.html', data=data, jobs=jobs, job_graphs=job_graphs)

@app.route('/compare', methods=['POST'])
def compare():
    if 'user_id' not in session or session.get('is_admin') == 0:
        return redirect(url_for('index'))
        
    compare_ids = request.form.getlist('compare_ids')
    if not compare_ids:
        return redirect(url_for('dashboard'))
        
    conn = get_db_connection()
    c = conn.cursor()
    placeholders = ','.join(['?'] * len(compare_ids))
    query = f"SELECT * FROM candidates WHERE id IN ({placeholders})"
    candidates = c.execute(query, compare_ids).fetchall()
    conn.close()
    
    return render_template('compare.html', candidates=candidates)

if __name__ == '__main__':
    app.run(debug=True, port=5000)