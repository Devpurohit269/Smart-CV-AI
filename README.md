# Smart CV AI — ATS & Skill Detection Upgrade

This package preserves the uploaded Flask interface and current resume-to-job comparison, and adds:
- Expanded canonical skill aliases and phrase-aware matching.
- An ATS-readiness checklist based on extracted text.
- Rule-based resume improvement suggestions.

## Important scope notes
- ATS checks are heuristics, not an official ATS score and not a layout simulator.
- Skill detection uses a curated dictionary and aliases; it is not a trained AI model.
- Resume suggestions are rule-based. No external AI API is called.
- SQLite authentication/history and sample job recommendations are not included in this first feature upgrade; they can be added in a later phase.
- The existing weighted resume match score remains a text/skill similarity estimate, not a hiring prediction.

## Windows PowerShell setup

Open PowerShell in the extracted `Smart_CV_AI_Upgraded` folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
py app.py
```

Then open `http://127.0.0.1:5000` in your browser.

If PowerShell blocks activation, you can skip activation and use the venv Python directly:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

## Project layout

- `app.py` — Flask routes, PDF/DOCX extraction, scoring, skill detection, ATS checks, suggestions
- `templates/index.html` — original upload interface
- `templates/result.html` — original results plus ATS and suggestions panels
- `static/style.css` — original styles plus the new panels
- `requirements.txt` — required Python packages

## Notes
- Maximum upload size remains 5 MB.
- Use text-based PDF files; scanned image-only PDFs may need OCR, which is not included.
- The development server runs with debug enabled locally. Do not expose it directly to the public internet; use a production WSGI server and proper secret/security configuration for deployment.
