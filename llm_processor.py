"""
llm_processor.py — Call OpenRouter LLM to extract structured JSON from document text.
Features: auto doc-type detection, confidence levels, evidence text, AI summary,
          missing/unclear field identification, AI review suggestions.
"""
 
import os
import json
import requests
from dotenv import load_dotenv
 
load_dotenv()
 
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-r1"
 
SCHEMA_BY_TYPE = {
    "Invoice": ["invoice_number", "invoice_date", "due_date", "vendor_name", "vendor_email",
                "vendor_phone", "vendor_address", "customer_name", "customer_email",
                "customer_address", "subtotal", "tax", "discount", "total_amount",
                "currency", "payment_terms", "notes"],
    "Receipt": ["receipt_number", "receipt_date", "merchant_name", "merchant_address",
                "merchant_phone", "subtotal", "tax", "total_amount", "payment_method", "currency"],
    "Admission Form": ["applicant_name", "date_of_birth", "gender", "email", "phone_number",
                       "address", "nationality", "program_applied", "application_date",
                       "application_id", "guardian_name", "guardian_contact"],
    "Certificate": ["certificate_type", "certificate_number", "recipient_name", "issue_date",
                    "expiry_date", "issuing_authority", "issuer_name", "issuer_designation",
                    "description", "awarded_for", "venue"],
    "General Form": ["form_title", "form_number", "submission_date", "submitter_name",
                     "submitter_email", "submitter_phone", "organization", "signature", "remarks"]
}
 
 
def _call_llm(prompt: str, max_tokens: int = 2000) -> str | None:
    if not OPENROUTER_API_KEY:
        return None
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://doc-extractor.local",
        "X-Title": "Document Intelligence Extractor"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": max_tokens
    }
    try:
        resp = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=90)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return None
 
 
def _parse_json(content: str) -> dict | list | None:
    if not content:
        return None
    # Strip markdown fences
    if "```" in content:
        parts = content.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except:
                continue
    try:
        return json.loads(content)
    except:
        # Try to find JSON object/array in content
        for start, end in [('{', '}'), ('[', ']')]:
            s = content.find(start)
            e = content.rfind(end)
            if s != -1 and e != -1:
                try:
                    return json.loads(content[s:e+1])
                except:
                    pass
    return None
 
 
def detect_document_type(raw_text: str) -> str:
    """Feature 1: Auto-detect document type from text."""
    prompt = f"""Analyze the following document text and determine its type.
Choose EXACTLY one from: Invoice, Receipt, Admission Form, Certificate, General Form
 
Rules:
- Return ONLY a JSON object with two keys: "doc_type" and "reasoning"
- doc_type must be one of the five options above exactly
- reasoning should be 1 sentence explaining why
 
Document Text:
\"\"\"
{raw_text[:3000]}
\"\"\"
 
Respond with JSON only:"""
 
    content = _call_llm(prompt, max_tokens=200)
    result = _parse_json(content) if content else None
    if result and "doc_type" in result:
        doc_type = result["doc_type"]
        if doc_type in SCHEMA_BY_TYPE:
            return doc_type, result.get("reasoning", "")
    return "General Form", "Could not determine document type automatically."
 
 
def extract_structured_data(raw_text: str, doc_type: str) -> dict:
    """Feature 2 & 3: Extract fields with confidence levels and evidence text."""
    if not OPENROUTER_API_KEY:
        return {"error": "OPENROUTER_API_KEY is not set in the .env file."}
 
    fields = SCHEMA_BY_TYPE.get(doc_type, SCHEMA_BY_TYPE["General Form"])
    fields_str = "\n".join(f'- "{f}"' for f in fields)
 
    prompt = f"""You are a precise document data extraction AI.
 
Document Type: {doc_type}
 
Extract the following fields from the document text below.
For EACH field, return a JSON object with:
- "value": the extracted value (null if not found)
- "confidence": "High", "Medium", or "Low"
  * High = value is explicitly and clearly stated in text
  * Medium = value is implied or partially stated
  * Low = value is inferred or uncertain
- "evidence": the exact short snippet of text (under 20 words) that supports this value, or null if not found
 
Fields to extract:
{fields_str}
 
Additional rules:
- Return ONLY a valid JSON object where each key is a field name and value is {{"value":..., "confidence":..., "evidence":...}}
- Dates must be YYYY-MM-DD format
- Monetary amounts must be numbers
- No markdown fences, no explanation
 
Document Text:
\"\"\"
{raw_text[:6000]}
\"\"\"
 
JSON response:"""
 
    content = _call_llm(prompt, max_tokens=3000)
    result = _parse_json(content)
    if result is None:
        return {"error": "LLM returned invalid JSON. Please try again."}
    return result
 
 
def generate_summary(raw_text: str, doc_type: str) -> str:
    """Feature 4: Generate a short AI summary of the document."""
    prompt = f"""You are a document analysis AI. Write a concise 2-3 sentence summary of the following {doc_type} document.
Focus on the key parties involved, the main purpose, and any important amounts or dates.
 
Document Text:
\"\"\"
{raw_text[:4000]}
\"\"\"
 
Summary (2-3 sentences only):"""
 
    content = _call_llm(prompt, max_tokens=300)
    return content if content else "Summary could not be generated."
 
 
def analyze_fields(extracted: dict, doc_type: str, raw_text: str) -> dict:
    """Features 5 & 6: Identify missing/unclear/incorrect fields and give AI review suggestions."""
    if not OPENROUTER_API_KEY:
        return {"missing": [], "unclear": [], "suggestions": []}
 
    # Build a simplified view of extracted data for the prompt
    simplified = {}
    for k, v in extracted.items():
        if isinstance(v, dict):
            simplified[k] = {"value": v.get("value"), "confidence": v.get("confidence")}
        else:
            simplified[k] = v
 
    prompt = f"""You are a document quality review AI analyzing a {doc_type}.
 
Here are the extracted fields and their confidence levels:
{json.dumps(simplified, indent=2)}
 
Document text snippet:
\"\"\"
{raw_text[:3000]}
\"\"\"
 
Analyze the extraction and return a JSON object with exactly these three keys:
1. "missing_fields": list of field names that are null/empty but should be present in this document type
2. "unclear_fields": list of field names where the value exists but confidence is Low or the value seems incorrect
3. "suggestions": list of specific, actionable suggestion strings to improve the document or extraction
   (e.g. "Invoice number appears to be missing - check the top-right header of the document")
 
Return ONLY valid JSON, no markdown fences:"""
 
    content = _call_llm(prompt, max_tokens=1000)
    result = _parse_json(content)
    if result and all(k in result for k in ["missing_fields", "unclear_fields", "suggestions"]):
        return result
    return {"missing_fields": [], "unclear_fields": [], "suggestions": ["Could not generate analysis."]}
