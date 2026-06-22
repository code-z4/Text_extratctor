"""
llm_processor.py — Call OpenRouter LLM to extract structured JSON from document text.
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
    "Invoice": {
        "invoice_number": "string",
        "invoice_date": "YYYY-MM-DD",
        "due_date": "YYYY-MM-DD",
        "vendor_name": "string",
        "vendor_email": "string",
        "vendor_phone": "string",
        "vendor_address": "string",
        "customer_name": "string",
        "customer_email": "string",
        "customer_address": "string",
        "line_items": "[{description, quantity, unit_price, total}]",
        "subtotal": "number",
        "tax": "number",
        "discount": "number",
        "total_amount": "number",
        "currency": "string",
        "payment_terms": "string",
        "notes": "string"
    },
    "Receipt": {
        "receipt_number": "string",
        "receipt_date": "YYYY-MM-DD",
        "merchant_name": "string",
        "merchant_address": "string",
        "merchant_phone": "string",
        "items_purchased": "[{item_name, quantity, price}]",
        "subtotal": "number",
        "tax": "number",
        "total_amount": "number",
        "payment_method": "string",
        "currency": "string"
    },
    "Admission Form": {
        "applicant_name": "string",
        "date_of_birth": "YYYY-MM-DD",
        "gender": "string",
        "email": "string",
        "phone_number": "string",
        "address": "string",
        "nationality": "string",
        "program_applied": "string",
        "academic_qualifications": "[{degree, institution, year, grade}]",
        "application_date": "YYYY-MM-DD",
        "application_id": "string",
        "guardian_name": "string",
        "guardian_contact": "string",
        "documents_submitted": "[string]"
    },
    "Certificate": {
        "certificate_type": "string",
        "certificate_number": "string",
        "recipient_name": "string",
        "issue_date": "YYYY-MM-DD",
        "expiry_date": "YYYY-MM-DD",
        "issuing_authority": "string",
        "issuer_name": "string",
        "issuer_designation": "string",
        "description": "string",
        "awarded_for": "string",
        "venue": "string"
    },
    "General Form": {
        "form_title": "string",
        "form_number": "string",
        "submission_date": "YYYY-MM-DD",
        "submitter_name": "string",
        "submitter_email": "string",
        "submitter_phone": "string",
        "organization": "string",
        "fields": "{key: value pairs of all form fields}",
        "signature": "string",
        "remarks": "string"
    }
}


def build_prompt(text: str, doc_type: str) -> str:
    schema = SCHEMA_BY_TYPE.get(doc_type, SCHEMA_BY_TYPE["General Form"])
    schema_str = json.dumps(schema, indent=2)

    return f"""You are a precise document data extraction AI. Extract structured information from the document text below.

Document Type: {doc_type}

Extract ONLY the following fields (use null for any field not found in the document):
{schema_str}

Rules:
- Return ONLY a valid JSON object. No markdown fences, no explanation, no extra text.
- Dates must be in YYYY-MM-DD format if found, otherwise null.
- Monetary amounts must be numbers (not strings), e.g. 1500.00
- Phone numbers should be in international format if possible.
- Email addresses must be valid-looking strings.
- For lists, return an empty array [] if no items found.

Document Text:
\"\"\"
{text[:6000]}
\"\"\"

Respond with the JSON object only:"""


def extract_structured_data(raw_text: str, doc_type: str) -> dict:
    if not OPENROUTER_API_KEY:
        return {"error": "OPENROUTER_API_KEY is not set in the .env file."}

    prompt = build_prompt(raw_text, doc_type)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://doc-extractor.local",
        "X-Title": "Document Intelligence Extractor"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 2000
    }

    try:
        response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"].strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        if content.endswith("```"):
            content = content[:-3].strip()

        return json.loads(content)

    except requests.exceptions.Timeout:
        return {"error": "Request to OpenRouter timed out. Please try again."}
    except requests.exceptions.HTTPError as e:
        return {"error": f"OpenRouter API error: {e.response.status_code} — {e.response.text[:200]}"}
    except json.JSONDecodeError as e:
        return {"error": f"LLM returned invalid JSON: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}
