import streamlit as st
import json
import os
from datetime import datetime
from dotenv import load_dotenv

from extractor import extract_text_from_pdf, extract_text_from_docx
from llm_processor import extract_structured_data
from validator import validate_document_data, get_review_status

load_dotenv()

st.set_page_config(
    page_title="Document Intelligence Extractor",
    page_icon="📄",
    layout="wide"
)

# --- Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #5a7fa8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .status-approved {
        background-color: #d4edda;
        color: #155724;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .status-review {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .status-rejected {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .metric-box {
        background: #f0f4f8;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e3a5f;
        border-bottom: 2px solid #e0e8f0;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-header">📄 Document Intelligence Extractor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a PDF or Word document, select a type, and extract structured JSON data using AI.</div>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    doc_type = st.selectbox(
        "Document Type",
        ["Invoice", "Receipt", "Admission Form", "Certificate", "General Form"],
        help="Select the type that best matches your document."
    )
    st.markdown("---")
    st.markdown("**Supported Formats**")
    st.markdown("- 📄 PDF (`.pdf`)")
    st.markdown("- 📝 Word (`.docx`)")
    st.markdown("---")
    st.markdown("**LLM Provider**")
    st.markdown("OpenRouter — `deepseek/deepseek-r1` (>12B)")

# --- File Upload ---
uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "docx"],
    help="Drag and drop or click to upload a PDF or Word document."
)

if uploaded_file:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.success(f"✅ File uploaded: **{uploaded_file.name}**")
        st.caption(f"Size: {uploaded_file.size / 1024:.1f} KB | Type: {doc_type}")

    if st.button("🚀 Extract & Analyse", use_container_width=True, type="primary"):
        with st.spinner("Extracting text from document…"):
            file_bytes = uploaded_file.read()
            filename = uploaded_file.name.lower()

            if filename.endswith(".pdf"):
                raw_text = extract_text_from_pdf(file_bytes)
            elif filename.endswith(".docx"):
                raw_text = extract_text_from_docx(file_bytes)
            else:
                st.error("Unsupported file format.")
                st.stop()

        if not raw_text or len(raw_text.strip()) < 20:
            st.error("Could not extract readable text from the document. Please try a different file.")
            st.stop()

        with st.expander("📃 Raw Extracted Text", expanded=False):
            st.text_area("", raw_text, height=200, label_visibility="collapsed")

        with st.spinner("Sending to LLM for structured extraction…"):
            result = extract_structured_data(raw_text, doc_type)

        if "error" in result:
            st.error(f"LLM extraction failed: {result['error']}")
            st.stop()

        # Validate
        validation_issues = validate_document_data(result, doc_type)
        review_status = get_review_status(validation_issues)

        # Status badge
        status_classes = {
            "Auto Approved": "status-approved",
            "Needs Review": "status-review",
            "Rejected": "status-rejected",
        }
        status_icons = {
            "Auto Approved": "✅",
            "Needs Review": "⚠️",
            "Rejected": "❌",
        }
        css_class = status_classes.get(review_status, "status-review")
        icon = status_icons.get(review_status, "⚠️")

        st.markdown("---")
        st.markdown(f"### Review Status")
        st.markdown(f'<span class="{css_class}">{icon} {review_status}</span>', unsafe_allow_html=True)

        if validation_issues:
            with st.expander("🔍 Validation Issues", expanded=(review_status == "Rejected")):
                for issue in validation_issues:
                    st.warning(f"• {issue}")
        else:
            st.success("All validation checks passed.")

        st.markdown("---")
        st.markdown('<div class="section-title">📊 Extracted Structured Data</div>', unsafe_allow_html=True)

        # Pretty field view
        tab1, tab2 = st.tabs(["🗂 Field View", "{ } Raw JSON"])

        with tab1:
            if isinstance(result, dict):
                for key, value in result.items():
                    cols = st.columns([1, 2])
                    cols[0].markdown(f"**{key.replace('_', ' ').title()}**")
                    if isinstance(value, (dict, list)):
                        cols[1].json(value)
                    else:
                        cols[1].markdown(str(value) if value else "_Not found_")
            else:
                st.json(result)

        with tab2:
            final_output = {
                "document_type": doc_type,
                "filename": uploaded_file.name,
                "extracted_at": datetime.utcnow().isoformat() + "Z",
                "review_status": review_status,
                "validation_issues": validation_issues,
                "extracted_data": result
            }
            st.json(final_output)
            json_str = json.dumps(final_output, indent=2)
            st.download_button(
                label="⬇️ Download JSON",
                data=json_str,
                file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_extracted.json",
                mime="application/json"
            )

else:
    st.info("👆 Upload a document to get started.")
    # Demo/sample section
    st.markdown("---")
    st.markdown("### 💡 What this tool does")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📤 Upload**\nPDF or Word documents are accepted.")
    with col2:
        st.markdown("**🤖 Extract**\nAI extracts key fields based on document type.")
    with col3:
        st.markdown("**✅ Validate**\nFields are validated and a review status is assigned.")
