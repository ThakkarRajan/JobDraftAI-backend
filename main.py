from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PyPDF2 import PdfReader
import openai
import os
import io
import json
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "https://job-draft-ai.vercel.app",
    "http://localhost:3000"  # for local dev
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Validate Resume PDF Endpoint
# ---------------------------
@app.post("/validate-resume")
async def validate_resume(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        reader = PdfReader(io.BytesIO(contents))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""

        keywords = ["resume", "experience", "skills", "education", "projects", "summary"]
        match_count = sum(1 for k in keywords if k in text.lower())

        return {"valid": match_count >= 3, "matches": match_count} if match_count >= 3 else {
            "valid": False,
            "message": "The uploaded PDF doesn't appear to be a resume.",
            "matches": match_count
        }
    except Exception as e:
        return {"valid": False, "message": f"Error reading PDF: {str(e)}"}

# ---------------------------
# Extract Resume Text
# ---------------------------
@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        reader = PdfReader(io.BytesIO(contents))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# AI Tailoring Endpoint
# ---------------------------
class ResumeText(BaseModel):
    text: str

@app.post("/process-text")
async def process_text(data: ResumeText):
    logging.info("📥 Processing resume text")

    prompt = f"""
You are an advanced AI resume assistant.

The user provides a raw resume and a job description. Your job is to **rewrite and tailor** the resume so it is **ATS-friendly** and strongly aligned with the job.

Focus on:
- Improving the professional summary to reflect the job
- Extracting and enhancing relevant skills
- Rewriting experience descriptions to match job responsibilities
- Adding **at least one certificate**
- Ensuring **each experience has 4 tailored bullet points**
- Ensuring **each project has at least 2 specific bullet points**
- If no experience is found, generate a relevant **volunteer or internship experience** based on the job
- Return only valid JSON in the structure below (no markdown, no code fences, no explanation)

Output Format (return this as pure JSON only):
{{
  "name": "...",
  "contact": {{
    "location": "...",
    "email": "...",
    "phone": "...",
    "website": "...",
    "github": "...",
    "linkedin": "..."
  }},
  "tailored_summary": "...",
  "tailored_skills": {{
    "Programming Languages": [...],
    "Frameworks": [...],
    "Api Development": [...],
    "Version Control": [...],
    "Other Skills": [...]
  }},
  "tailored_experience": [
    {{
      "company": "...",
      "title": "...",
      "location": "...",
      "start": "...",
      "end": "...",
      "highlights": ["...", "...", "...", "..."]
    }}
  ],
  "tailored_certificates": ["..."],
  "projects": [
    {{
      "title": "...",
      "tech": ["..."],
      "description": "...",
      "highlights": ["...", "..."]
    }}
  ],
  "education": [
    {{
      "program": "...",
      "school": "...",
      "location": "...",
      "start": "...",
      "end": "..."
    }}
  ]
}}

Resume and job description:
{data.text}
"""


    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful resume assistant that returns JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        raw_content = response.choices[0].message.content

        # Parse JSON safely
        json_start = raw_content.find("{")
        json_end = raw_content.rfind("}") + 1
        json_string = raw_content[json_start:json_end]

        try:
            structured = json.loads(json_string)
            return {"structured": structured}
        except json.JSONDecodeError:
            logging.error("❌ JSON decode failed")
            raise HTTPException(status_code=500, detail="Failed to parse JSON response.")

    except Exception as e:
        logging.error("❌ OpenAI API error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# Run the App
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)