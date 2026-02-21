import sqlite3
import pandas as pd
from ml.resume_parser import match_resume
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def generate_ranking_graph(db_path, output_path, job_id=None):
    conn = sqlite3.connect(db_path)
    
    # Extract candidate metrics
    query = "SELECT name, skills, quiz, experience, summary FROM candidates"
    
    try:
        df = pd.read_sql_query(query, conn)
        
        job_desc = ""
        if job_id is not None:
            job_query = f"SELECT description, skills_required FROM jobs WHERE id={job_id}"
            job_row = pd.read_sql_query(job_query, conn)
            if not job_row.empty:
                job_desc = f"{job_row['description'].iloc[0]} {job_row['skills_required'].iloc[0]}"
                
    except Exception as e:
        print("Error reading from db:", e)
        conn.close()
        return False
        
    conn.close()
    
    if df.empty:
        return False

    if job_id is not None and job_desc:
        # Calculate dynamic TF-IDF against this specific job
        df['resume_match'] = df['summary'].fillna("").apply(lambda x: match_resume(x, job_desc))
    else:
        # Fallback if no specific job, just use their existing skills string as a weak generic match
        df['resume_match'] = df['summary'].fillna("").apply(lambda x: match_resume(x, "software engineering machine learning nlp web application"))
    
    # Normalize skills and experience if needed (making out of 100 roughly)
    df['exp_norm'] = df['experience'].apply(lambda x: min(100, float(x) * 10))
    df['skills_norm'] = df['skills'].apply(lambda x: min(100, float(x) * 10))
    df['quiz'] = df['quiz'].fillna(0)
    
    # Weighted final score: 40% resume, 30% quiz, 20% skills, 10% experience
    df['final_score'] = (0.40 * df['resume_match']) + (0.30 * df['quiz']) + (0.20 * df['skills_norm']) + (0.10 * df['exp_norm'])
    
    df = df.groupby('name')['final_score'].max().reset_index()
    df = df.sort_values(by='final_score', ascending=True)
    
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    
    # Build bars
    bars = plt.barh(df['name'], df['final_score'], color='#818cf8', height=0.6)
    
    # Formats
    title_text = f'Candidate Leaderboard (Job #{job_id})' if job_id else 'Overall Candidate Leaderboard'
    plt.xlabel('Final Analytical Score (0-100)', fontsize=12, color='#94a3b8', labelpad=10)
    plt.title(title_text, fontsize=16, color='#f8fafc', pad=20, fontweight='bold')
    plt.xlim(0, 100)
    
    ax.set_facecolor('none')
    plt.gcf().patch.set_facecolor('none')
    ax.spines['bottom'].set_color('#334155')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.tick_params(axis='x', colors='#94a3b8', labelsize=10)
    ax.tick_params(axis='y', colors='#e2e8f0', labelsize=12)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 2, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                 ha='left', va='center', color='#34d399', fontweight='bold', fontsize=11)
                 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', transparent=True)
    plt.close()
    
    return True
