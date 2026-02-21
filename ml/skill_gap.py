def skill_gap(candidate_skills, required_job_skills=None):
    if required_job_skills:
        job_skills = [s.strip().lower() for s in required_job_skills.split(",")]
    else:
        job_skills = ["python", "sql", "react", "docker", "machine learning"]

    missing = []
    
    cand_skills_lower = candidate_skills.lower() if candidate_skills else ""

    for s in job_skills:
        if s not in cand_skills_lower:
            missing.append(s)

    return missing
