"""
validator.py — Validate extracted document fields and assign a review status.
"""

import re
from typing import List


# ── Regex patterns ────────────────────────────────────────────────────────────
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
EMAIL_RE = re.compile(r"^[\w.+\-]+@[\w\-]+\.[a-zA-Z]{2,}$")
PHONE_RE = re.compile(r"^[\+\d\s\-\(\)]{7,20}$")


def _is_valid_date(value) -> bool:
    if not isinstance(value, str):
        return False
    if not DATE_RE.match(value):
        return False
    try:
        from datetime import date
        year, month, day = map(int, value.split("-"))
        date(year, month, day)
        return True
    except ValueError:
        return False


def _is_valid_email(value) -> bool:
    return isinstance(value, str) and bool(EMAIL_RE.match(value))


def _is_valid_phone(value) -> bool:
    return isinstance(value, str) and bool(PHONE_RE.match(value.strip()))


def _is_positive_amount(value) -> bool:
    try:
        return float(value) >= 0
    except (TypeError, ValueError):
        return False


# ── Field rules per document type ────────────────────────────────────────────
REQUIRED_FIELDS = {
    "Invoice": ["invoice_number", "invoice_date", "vendor_name", "total_amount"],
    "Receipt": ["receipt_date", "merchant_name", "total_amount"],
    "Admission Form": ["applicant_name", "email", "program_applied", "application_date"],
    "Certificate": ["recipient_name", "issue_date", "issuing_authority"],
    "General Form": ["form_title", "submitter_name"],
}

DATE_FIELDS = {
    "Invoice": ["invoice_date", "due_date"],
    "Receipt": ["receipt_date"],
    "Admission Form": ["date_of_birth", "application_date"],
    "Certificate": ["issue_date", "expiry_date"],
    "General Form": ["submission_date"],
}

EMAIL_FIELDS = {
    "Invoice": ["vendor_email", "customer_email"],
    "Receipt": [],
    "Admission Form": ["email"],
    "Certificate": [],
    "General Form": ["submitter_email"],
}

PHONE_FIELDS = {
    "Invoice": ["vendor_phone"],
    "Receipt": ["merchant_phone"],
    "Admission Form": ["phone_number"],
    "Certificate": [],
    "General Form": ["submitter_phone"],
}

AMOUNT_FIELDS = {
    "Invoice": ["subtotal", "total_amount", "tax"],
    "Receipt": ["subtotal", "total_amount", "tax"],
    "Admission Form": [],
    "Certificate": [],
    "General Form": [],
}


def validate_document_data(data: dict, doc_type: str) -> List[str]:
    """
    Run all validation checks and return a list of issue strings.
    Empty list means no issues found.
    """
    issues: List[str] = []

    if not isinstance(data, dict):
        return ["Extracted data is not a valid JSON object."]

    # 1. Required fields
    for field in REQUIRED_FIELDS.get(doc_type, []):
        value = data.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            issues.append(f"Required field missing or empty: '{field}'")

    # 2. Date format
    for field in DATE_FIELDS.get(doc_type, []):
        value = data.get(field)
        if value and not _is_valid_date(value):
            issues.append(f"Invalid date format for '{field}': expected YYYY-MM-DD, got '{value}'")

    # 3. Email format
    for field in EMAIL_FIELDS.get(doc_type, []):
        value = data.get(field)
        if value and not _is_valid_email(value):
            issues.append(f"Invalid email address in '{field}': '{value}'")

    # 4. Phone format
    for field in PHONE_FIELDS.get(doc_type, []):
        value = data.get(field)
        if value and not _is_valid_phone(value):
            issues.append(f"Invalid phone number in '{field}': '{value}'")

    # 5. Amount fields
    for field in AMOUNT_FIELDS.get(doc_type, []):
        value = data.get(field)
        if value is not None and not _is_positive_amount(value):
            issues.append(f"Invalid amount in '{field}': must be a non-negative number, got '{value}'")

    return issues


def get_review_status(issues: List[str]) -> str:
    """
    Assign review status based on the number and type of validation issues.
    """
    if not issues:
        return "Auto Approved"

    critical_keywords = ["Required field missing", "Invalid date format"]
    critical_issues = [i for i in issues if any(k in i for k in critical_keywords)]

    if len(critical_issues) >= 2 or len(issues) >= 4:
        return "Rejected"

    return "Needs Review"
