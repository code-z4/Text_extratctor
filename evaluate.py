"""
evaluate.py — Evaluate the extraction pipeline with sample document texts.
Run: python evaluate.py
"""

import json
import os
from dotenv import load_dotenv
from llm_processor import extract_structured_data
from validator import validate_document_data, get_review_status

load_dotenv()

SAMPLE_DOCUMENTS = [
    {
        "name": "Sample Invoice",
        "doc_type": "Invoice",
        "text": """
        INVOICE

        Invoice Number: INV-2024-0042
        Invoice Date: 2024-05-15
        Due Date: 2024-06-15

        From:
        TechSolutions Pvt. Ltd.
        123 MG Road, Bengaluru, Karnataka 560001
        Email: billing@techsolutions.in
        Phone: +91 98765 43210

        To:
        GlobalCorp India Ltd.
        456 Nariman Point, Mumbai, Maharashtra 400021
        Email: accounts@globalcorp.com

        Items:
        1. Web Development Services  -  40 hrs @ ₹2,500  =  ₹1,00,000
        2. UI/UX Design              -  20 hrs @ ₹2,000  =  ₹40,000
        3. Cloud Deployment Setup    -  1 unit @ ₹15,000 =  ₹15,000

        Subtotal:  ₹1,55,000
        GST (18%): ₹27,900
        Total:     ₹1,82,900

        Payment Terms: Net 30 days
        Notes: Please transfer to HDFC Account No. XXXX-XXXX-1234
        """
    },
    {
        "name": "Sample Receipt",
        "doc_type": "Receipt",
        "text": """
        RECEIPT
        
        Receipt No: RCP-9910
        Date: 2024-03-22
        
        Store: FreshMart Superstore
        Address: 78 Park Street, Kolkata - 700016
        Phone: 033-2229-8800
        
        Items:
        - Basmati Rice 5kg     x1   Rs. 450
        - Toor Dal 1kg         x2   Rs. 280
        - Sunflower Oil 1L     x2   Rs. 320
        - Amul Butter 500g     x1   Rs. 275
        
        Subtotal:  Rs. 1,325
        Tax (5%):  Rs. 66.25
        TOTAL:     Rs. 1,391.25
        
        Payment: UPI (Google Pay)
        Thank you for shopping with us!
        """
    },
    {
        "name": "Sample Certificate",
        "doc_type": "Certificate",
        "text": """
        CERTIFICATE OF COMPLETION

        This is to certify that

        MS. PRIYA SHARMA

        has successfully completed the course

        "Advanced Python Programming & Data Science"

        Conducted by: DataLearn Academy, New Delhi
        Duration: 3 months (January 2024 – March 2024)
        Grade Obtained: A+ (95%)

        Date of Issue: 2024-04-01

        Awarded by:
        Dr. Rajesh Kumar
        Director, DataLearn Academy
        Certificate No: DLA-PY-2024-0157
        """
    },
    {
        "name": "Incomplete/Problematic Form",
        "doc_type": "General Form",
        "text": """
        FEEDBACK FORM

        Name: Aakash
        Email: not-an-email
        Phone: 12345
        Date: 32-13-2024
        Comments: The service was okay.
        Rating: 4/5
        """
    }
]


def run_evaluation():
    print("=" * 70)
    print(" DOCUMENT EXTRACTION EVALUATION REPORT")
    print("=" * 70)
    print(f"Model: deepseek/deepseek-r1 via OpenRouter")
    print(f"Total samples: {len(SAMPLE_DOCUMENTS)}")
    print("=" * 70)

    results_summary = []

    for i, sample in enumerate(SAMPLE_DOCUMENTS, 1):
        print(f"\n[{i}/{len(SAMPLE_DOCUMENTS)}] {sample['name']} ({sample['doc_type']})")
        print("-" * 50)

        extracted = extract_structured_data(sample["text"], sample["doc_type"])
        issues = validate_document_data(extracted, sample["doc_type"])
        status = get_review_status(issues)

        print(f"  ✔ Extraction: {'SUCCESS' if 'error' not in extracted else 'FAILED'}")
        if "error" in extracted:
            print(f"    Error: {extracted['error']}")
        else:
            non_null = sum(1 for v in extracted.values() if v is not None and v != "" and v != [] and v != {})
            print(f"  ✔ Fields extracted (non-null): {non_null}/{len(extracted)}")

        print(f"  ✔ Review Status: {status}")
        if issues:
            print(f"  ⚠ Validation Issues ({len(issues)}):")
            for issue in issues:
                print(f"      - {issue}")
        else:
            print("  ✔ No validation issues.")

        results_summary.append({
            "sample": sample["name"],
            "doc_type": sample["doc_type"],
            "extraction_success": "error" not in extracted,
            "review_status": status,
            "validation_issues": issues,
            "extracted_fields": extracted if "error" not in extracted else {}
        })

    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    successful = sum(1 for r in results_summary if r["extraction_success"])
    print(f"  Extraction Success Rate : {successful}/{len(results_summary)}")
    status_counts = {}
    for r in results_summary:
        status_counts[r["review_status"]] = status_counts.get(r["review_status"], 0) + 1
    for status, count in status_counts.items():
        print(f"  {status:<20}: {count} document(s)")
    print("=" * 70)

    # Save report
    report_path = "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
    print(f"\n📄 Full report saved to: {report_path}")


if __name__ == "__main__":
    run_evaluation()
