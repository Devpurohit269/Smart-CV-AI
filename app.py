from __future__ import annotations

import re
from io import BytesIO
from typing import Any

from flask import Flask, render_template, request
import pymupdf
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# Canonical skill names and common spellings/aliases.
# This remains a curated dictionary, not a trained NLP model.
SKILL_ALIASES = {
    "HTML": ["html", "html5"],
    "CSS": ["css", "css3", "cascading style sheets"],
    "JavaScript": ["javascript", "java script", "js"],
    "TypeScript": ["typescript", "ts"],
    "React": ["react", "react.js", "reactjs"],
    "Next.js": ["next.js", "nextjs"],
    "Vue.js": ["vue", "vue.js", "vuejs"],
    "Angular": ["angular", "angularjs"],
    "Bootstrap": ["bootstrap"],
    "Tailwind CSS": ["tailwind css", "tailwindcss"],
    "Responsive Design": ["responsive design", "mobile-first design"],
    "Flexbox": ["flexbox", "css flexbox"],
    "CSS Grid": ["css grid", "grid layout"],
    "DOM Manipulation": ["dom manipulation", "document object model"],
    "REST API": ["rest api", "restful api", "rest apis", "restful apis"],
    "API Integration": ["api integration", "integrating apis"],
    "JSON": ["json", "javascript object notation"],
    "Git": ["git"],
    "GitHub": ["github"],
    "Version Control": ["version control", "source control"],
    "Vite": ["vite"],
    "npm": ["npm", "node package manager"],
    "Python": ["python", "python3"],
    "Java": ["java"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "c sharp"],
    "C": ["c programming", "programming in c"],
    "SQL": ["sql", "structured query language"],
    "Flask": ["flask"],
    "Django": ["django"],
    "FastAPI": ["fastapi", "fast api"],
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning"],
    "Data Analysis": ["data analysis", "data analytics"],
    "Pandas": ["pandas"],
    "NumPy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch", "torch"],
    "Problem Solving": ["problem solving", "problem-solving"],
    "Communication": ["communication", "communication skills"],
    "Figma": ["figma"],
    "UI/UX": ["ui/ux", "ui ux", "user interface design", "user experience design"],
    "Accessibility": ["accessibility", "web accessibility", "wcag"],
    "Testing": ["testing", "software testing"],
    "Debugging": ["debugging", "debug"],
    "Web Development": ["web development", "web dev"],
    "Web Design": ["web design"],
    "Node.js": ["node.js", "nodejs", "node js"],
    "Express.js": ["express.js", "expressjs", "express js"],
    "MongoDB": ["mongodb", "mongo db"],
    "MySQL": ["mysql", "my sql"],
    "PostgreSQL": ["postgresql", "postgres"],
    "SQLite": ["sqlite"],
    "Firebase": ["firebase"],
    "Redux": ["redux"],
    "Material UI": ["material ui", "mui"],
    "Sass": ["sass", "scss"],
    "Less": ["less css"],
    "Webpack": ["webpack"],
    "Agile": ["agile", "agile methodology"],
    "OOP": ["oop", "object-oriented programming", "object oriented programming"],
    "Data Structures": ["data structures", "data structure"],
    "Algorithms": ["algorithms", "algorithm design"],
    "SEO": ["seo", "search engine optimization"],
    "Performance Optimization": ["performance optimization", "web performance"],
    "Cross-browser Compatibility": ["cross-browser compatibility", "cross browser compatibility"],
    "Software Engineering": ["software engineering"],
    "Operating Systems": ["operating systems", "operating system"],
    "Computer Networks": ["computer networks", "networking"],
    "DBMS": ["dbms", "database management systems", "database management system"],
    "Unit Testing": ["unit testing", "unit tests"],
    "Data Structures and Algorithms": ["data structures and algorithms", "dsa"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Linux": ["linux"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure", "microsoft azure"],
    "Google Cloud": ["google cloud", "gcp"],
    "REST": ["rest architecture", "representational state transfer"],
    "GraphQL": ["graphql", "graph ql"],
    "PHP": ["php"],
    "Laravel": ["laravel"],
    "Ruby": ["ruby"],
    "Ruby on Rails": ["ruby on rails", "rails"],
    "Go": ["golang", "go language"],
    "Rust": ["rust"],
    "Power BI": ["power bi", "powerbi"],
    "Tableau": ["tableau"],
    "Excel": ["microsoft excel", "excel"],
    "Jupyter": ["jupyter", "jupyter notebook"],
}

# Kept as a simple list for compatibility with the original project's concept.
SKILLS = sorted(SKILL_ALIASES.keys())

STOP_WORDS = {
    "the", "and", "for", "with", "from", "this", "that", "are", "was",
    "were", "will", "have", "has", "had", "you", "your", "our", "their",
    "they", "them", "who", "what", "when", "where", "how", "can", "could",
    "should", "would", "into", "onto", "about", "over", "under", "using",
    "use", "used", "work", "working", "job", "role", "team", "experience",
    "knowledge", "ability", "strong", "good", "required", "requirements",
    "responsibilities", "developer", "development", "looking", "seeking",
    "candidate", "must", "able", "preferred", "including", "such", "other",
    "within", "across", "ensure", "support", "build", "create", "these",
    "those", "also", "more", "some", "all"
}


def normalize_text(text: str) -> str:
    """Normalize whitespace and common punctuation variants."""
    text = (text or "").lower()
    text = text.replace("\u2010", "-").replace("\u2011", "-")
    text = text.replace("\u2012", "-").replace("\u2013", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _alias_pattern(alias: str) -> re.Pattern:
    """Compile a phrase-aware regex; whitespace between words may vary."""
    pieces = re.split(r"\s+", normalize_text(alias))
    body = r"\s+".join(re.escape(piece) for piece in pieces)
    # Boundaries avoid matching 'java' inside 'javascript' and 'c' inside words.
    return re.compile(r"(?<![a-z0-9])" + body + r"(?![a-z0-9])", re.IGNORECASE)


_COMPILED_SKILL_PATTERNS = {
    canonical: [_alias_pattern(alias) for alias in aliases]
    for canonical, aliases in SKILL_ALIASES.items()
}


def extract_text(file) -> str:
    """Extract text from a text-based PDF or DOCX resume."""
    filename = (file.filename or "").lower()
    if filename.endswith(".pdf"):
        file_bytes = file.read()
        with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
            return "\n".join(page.get_text("text") for page in pdf)
    if filename.endswith(".docx"):
        document = Document(file)
        parts = [p.text for p in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    parts.append(cell.text)
        return "\n".join(parts)
    raise ValueError("Unsupported format. Upload a PDF or DOCX resume.")


def detect_skills(text: str) -> list[str]:
    """Return canonical skill labels using phrase-aware aliases."""
    normalized = normalize_text(text)
    found = [
        canonical
        for canonical, patterns in _COMPILED_SKILL_PATTERNS.items()
        if any(pattern.search(normalized) for pattern in patterns)
    ]
    # Keep the parent concept if DSA was detected, without duplicate aliases.
    return sorted(set(found), key=str.casefold)


def extract_keywords(text: str) -> set[str]:
    normalized = normalize_text(text)
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.]*", normalized)
    return {w for w in words if len(w) > 1 and w not in STOP_WORDS}


def calculate_similarity(resume_text: str, job_description: str) -> float:
    """Weighted comparison score out of 10 (not an official ATS score)."""
    if not resume_text.strip() or not job_description.strip():
        return 0.0

    resume_skills = set(detect_skills(resume_text))
    job_skills = set(detect_skills(job_description))
    skill_coverage = (
        len(resume_skills & job_skills) / len(job_skills)
        if job_skills else 0.0
    )

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        vectors = vectorizer.fit_transform([resume_text, job_description])
        text_similarity = float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])
    except ValueError:
        text_similarity = 0.0

    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_description)
    keyword_coverage = (
        len(resume_keywords & job_keywords) / len(job_keywords)
        if job_keywords else 0.0
    )

    score = 10 * (
        0.60 * skill_coverage
        + 0.25 * text_similarity
        + 0.15 * keyword_coverage
    )
    return round(min(10.0, max(0.0, score)), 1)


def run_ats_checks(resume_text: str, filename: str) -> dict[str, Any]:
    """
    Heuristic ATS-readiness checks. These inspect extracted text and do not
    simulate a specific employer's ATS or evaluate the original visual layout.
    """
    text = resume_text or ""
    lower = text.lower()
    words = re.findall(r"\b[\w+#.-]+\b", text)
    checks = []

    def add(title: str, passed: bool, detail: str, severity: str = "warning"):
        checks.append({
            "title": title,
            "status": "pass" if passed else severity,
            "label": "Looks good" if passed else ("Review" if severity == "warning" else "Info"),
            "detail": detail,
        })

    has_email = bool(re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, re.I))
    has_phone = bool(re.search(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)", text))
    add("Email address", has_email, "An email-like address was found." if has_email else "Add a professional email address near the top of your resume.")
    add("Phone number", has_phone, "A phone-like number was found." if has_phone else "Add a reachable phone number if appropriate for your applications.")

    headings = {
        "Experience": [r"\bwork experience\b", r"\bprofessional experience\b", r"\bemployment history\b", r"\bexperience\b"],
        "Education": [r"\beducation\b", r"\bacademic background\b"],
        "Skills": [r"\bskills\b", r"\btechnical skills\b", r"\bcore competencies\b"],
    }
    for title, patterns in headings.items():
        passed = any(re.search(p, lower) for p in patterns)
        add(f"Recognizable {title} section", passed,
            f"A recognizable {title.lower()} heading was found." if passed else f"Consider adding a clearly labeled '{title}' section when relevant.")

    has_metrics = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:%|percent|users|customers|projects|hours|days|weeks|months|years|x)\b", lower))
    add("Measurable achievements", has_metrics,
        "At least one number or metric may describe an outcome." if has_metrics else "Where truthful, add concrete outcomes (for example, time saved, users supported, or performance improved).")

    word_count = len(words)
    sensible_length = 150 <= word_count <= 1200
    add("Resume text length", sensible_length,
        f"About {word_count} words were extracted. This is within a broad review range." if sensible_length else f"About {word_count} words were extracted. Review whether the resume is too short or unnecessarily long; suitable length depends on experience.",
        severity="info")

    supported_file = filename.lower().endswith((".pdf", ".docx"))
    add("Supported document type", supported_file,
        "PDF/DOCX text extraction was used." if supported_file else "Use a text-based PDF or DOCX file.")

    # This is only a weak indicator, not a visual layout parser.
    has_sections = len(re.findall(r"(?im)^\s*(?:experience|education|skills|projects|summary|certifications)\s*$", text)) >= 2
    add("Section structure in extracted text", has_sections,
        "Several common section headings appear on separate lines." if has_sections else "Use simple, clearly labeled sections. Extraction text alone cannot verify columns, tables, icons, or visual reading order.",
        severity="info")

    passed_count = sum(item["status"] == "pass" for item in checks)
    score = round(100 * passed_count / len(checks))
    return {"score": score, "checks": checks, "passed": passed_count, "total": len(checks)}


def build_suggestions(
    resume_text: str,
    resume_skills: list[str],
    missing_skills: list[str],
    ats_report: dict[str, Any],
) -> list[str]:
    """Generate transparent, rule-based suggestions from the current analysis."""
    suggestions = []
    text = resume_text or ""
    lower = text.lower()

    if missing_skills:
        suggestions.append(
            "Compare the missing job skills with your actual experience. Add a skill only if you can support it with a project, course, or work example: "
            + ", ".join(missing_skills[:8]) + ("…" if len(missing_skills) > 8 else "") + "."
        )
    if not re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, re.I):
        suggestions.append("Add a professional email address in the contact section.")
    if not re.search(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)", text):
        suggestions.append("Consider adding a phone number with country code if you want recruiters to contact you by phone.")
    if not re.search(r"\b(?:experience|employment history|work experience|education|skills|projects)\b", lower):
        suggestions.append("Add clear section headings such as Summary, Skills, Projects, Education, and Experience where relevant.")
    if not re.search(r"\b\d+(?:\.\d+)?\s*(?:%|percent|users|customers|projects|hours|days|weeks|months|years|x)\b", lower):
        suggestions.append("Rewrite suitable project or work bullets to show truthful, measurable outcomes; do not invent numbers.")
    weak_phrases = ["responsible for", "worked on", "helped with", "duties included"]
    found_weak = [phrase for phrase in weak_phrases if phrase in lower]
    if found_weak:
        suggestions.append("Replace vague wording such as " + ", ".join(f'"{p}"' for p in found_weak) + " with a specific action and outcome where accurate.")
    if len(re.findall(r"\b[\w+#.-]+\b", text)) < 150:
        suggestions.append("The extracted resume is quite short. If relevant, include project context, your contribution, tools used, and outcomes.")
    if not suggestions:
        suggestions.append("Review each bullet for clarity, evidence, and relevance to the target role. Keep all claims accurate and verifiable.")
    return suggestions[:8]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    resume = request.files.get("resume")
    job_description = request.form.get("job_description", "").strip()

    if not resume or not resume.filename:
        return render_template("index.html", error="Please upload your resume.")
    filename = resume.filename.lower()
    if not filename.endswith((".pdf", ".docx")):
        return render_template("index.html", error="Please upload a PDF or DOCX file.")
    if not job_description:
        return render_template("index.html", error="Please enter a job description.")

    try:
        resume_text = extract_text(resume)
        if not resume_text.strip():
            return render_template(
                "index.html",
                error="No readable text was found. Try a text-based PDF or DOCX resume."
            )

        resume_skills = detect_skills(resume_text)
        job_skills = detect_skills(job_description)
        matched_skills = sorted(set(resume_skills) & set(job_skills), key=str.casefold)
        missing_skills = sorted(set(job_skills) - set(resume_skills), key=str.casefold)
        score = calculate_similarity(resume_text, job_description)
        ats_report = run_ats_checks(resume_text, filename)
        suggestions = build_suggestions(resume_text, resume_skills, missing_skills, ats_report)

        return render_template(
            "result.html",
            score=score,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            resume_skills=resume_skills,
            ats_report=ats_report,
            suggestions=suggestions,
        )
    except Exception:
        app.logger.exception("Resume analysis failed")
        return render_template(
            "index.html",
            error="We could not analyze this resume. Please check the file and try again."
        )


@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        error="Your upload is too large. The maximum size is 5 MB."
    ), 413


if __name__ == "__main__":
    # Local development only. Use a production WSGI server for deployment.
    app.run(debug=True)
