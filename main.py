from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PyPDF2 import PdfReader
from openai import OpenAI
import os
import io
import json
from dotenv import load_dotenv
import logging

# Load .env
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://job-draft-ai.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResumeText(BaseModel):
    text: str

@app.post("/validate-resume")
async def validate_resume(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        reader = PdfReader(io.BytesIO(contents))
        text = "".join(page.extract_text() or "" for page in reader.pages)

        keywords = ["resume", "experience", "skills", "education", "projects", "summary"]
        match_count = sum(1 for k in keywords if k in text.lower())

        return {"valid": match_count >= 3, "matches": match_count} if match_count >= 3 else {
            "valid": False,
            "message": "The uploaded PDF doesn't appear to be a resume.",
            "matches": match_count
        }
    except Exception as e:
        return {"valid": False, "message": f"Error reading PDF: {str(e)}"}

@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        reader = PdfReader(io.BytesIO(contents))
        text = "".join(page.extract_text() or "" for page in reader.pages)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-text")
async def process_text(data: ResumeText):
    prompt = f"""
You are an AI resume assistant. The user has provided a resume and a job description. Make it ATS-friendly and tailor it by improving the summary, highlighting relevant skills, rewriting experience highlights to match the role, and refining certifications.

Only return a valid JSON object with:
- name
- contact (location, email, phone, website, github, linkedin)
- tailored_summary
- tailored_skills (categorized)
- tailored_experience (company, title, location, start, end, highlights)
- tailored_certificates
- projects (title, tech, description)
- education (program, school, location, start, end)

Here is the resume and job context:
{data.text}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful resume assistant that only returns structured JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        content = response.choices[0].message.content
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        json_data = content[json_start:json_end]

        try:
            structured = json.loads(json_data)
            return {"structured": structured}
        except json.JSONDecodeError:
            logging.error("JSON decoding failed.")
            raise HTTPException(status_code=500, detail="AI returned invalid JSON.")
    except Exception as e:
        logging.error(f"OpenAI error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
