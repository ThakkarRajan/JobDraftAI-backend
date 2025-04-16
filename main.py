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
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

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
You are an AI resume assistant. The user provides a resume and a job description. Your task is to make it ATS friendly and tailor the resume and  to better fit the job by improving the summary, emphasizing relevant skills, updating experience explaination according to job profile, and surfacing matching certificates. Check that existing certification and experience highlights and project description is relevant to the job profile if not you can remove it and add appropriate data. If no data provided to the user data generate on your own that tailor to the job profile 

Return only a valid JSON object with this structure:
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
      "highlights": ["...", "..."]
    }}
  ],
  "tailored_certificates": ["..."],
  "projects": [
    {{
      "title": "...",
      "tech": ["..."],
      "description": "..."
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

Use the information below:
{data.text}

DO NOT return markdown, explanation, or wrap the JSON in triple backticks.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful resume assistant that outputs only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        raw_response = response["choices"][0]["message"]["content"]

        # Try to parse the JSON
        json_start = raw_response.find("{")
        json_end = raw_response.rfind("}") + 1
        json_string = raw_response[json_start:json_end]

        try:
            structured = json.loads(json_string)
            return {"structured": structured}
        except json.JSONDecodeError:
            logging.error("❌ Failed to decode AI response as JSON")
            raise HTTPException(status_code=500, detail="AI did not return valid JSON")

    except Exception as e:
        logging.error("❌ OpenAI error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"OpenAI error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
