# Publish to GitHub

Create a **public** repository named `DS-340W-Project` under your GitHub account.
Do not create a README or .gitignore on GitHub; this folder already contains them.

From inside the project folder, run:

```bash
git init -b main
git add .
git commit -m "Initial Market Relationship Project prototype"
git remote add origin https://github.com/arihaanbarua/DS-340W-Project.git
git push -u origin main
```

To verify, open `https://github.com/arihaanbarua/DS-340W-Project`.
Before sharing with class, ensure the repository is public.

Demo for Week 10: run `python -m pytest -q`, `python sample_run.py`, and `streamlit run app.py`. These tests verify an educational prototype on synthetic data, not replication of the original papers' published numerical results. The course's parent-paper replication requirement still needs separate evidence and a recording.
