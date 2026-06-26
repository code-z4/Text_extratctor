import streamlit as st
import json
import os
from datetime import datetime
from dotenv import load_dotenv
 
from extractor import extract_text_from_pdf, extract_text_from_docx
from llm_processor import detect_document_type, extract_structured_data, generate_summary, analyze_fields
from validator import validate_document_data, get_review_status
 
load_dotenv()
 
st.set_page_config(
    page_title="Document Intelligence Extractor",
    page_icon="",
    layout="wide"
)
 
st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1e3a5f; margin-bottom: 0.25rem; }
    .sub-header { color: #5a7fa8; font-size: 1rem; margin-bottom: 2rem; }
    .status-approved { background-color: #d4edda; color: #155724; padding: 0.4rem 1rem; border-radius: 20px; font-weight: 600; display: inline-block; }
    .status-review { background-color: #fff3cd; color: #856404; padding: 0.4rem 1rem; border-radius: 20px; font-weight: 600; display: inline-block; }
    .status-rejected { background-color: #f8d7da; color: #721c24; padding: 0.4rem 1rem; border-radius: 20px; font-weight: 600; display: inline-block; }
    .confidence-high { color: #155724; font-weight: 600; }
    .confidence-medium { color: #856404; font-weight: 600; }
    .confidence-low { color: #721c24; font-weight: 600; }
    .evidence-text { color: #6c757d; font-size: 0.85rem; font-style: italic; }
    .section-title { font-size: 1.1rem; font-weight: 600; color: #1e3a5f; border-bottom: 2px solid #e0e8f0; padding-bottom: 0.4rem; margin-bottom: 1rem; }
    .summary-box { background: #f0f4f8; border-left: 4px solid #1e3a5f; padding: 1rem 1.2rem; border-radius: 6px; margin-bottom: 1rem; }
    .detected-type { background: #e8f4fd; border: 1px solid #bee3f8; padding: 0.5rem 1rem; border-radius: 8px; display: inline-block; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)
 
st.markdown('<div class="main-header">Document Intelligence Extractor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a PDF or Word document. AI will detect the type, extract structured data, and provide a quality review.</div>', unsafe_allow_html=True)
 
with st.sidebar:
    st.header("Configuration")
    auto_detect = st.toggle("Auto-detect document type", value=True)
    if not auto_detect:
        doc_type = st.selectbox("Document Type", ["Invoice", "Receipt", "Admission Form", "Certificate", "General Form"])
    else:
        doc_type = None
    st.markdown("---")
    st.markdown("**Supported Formats**")
    st.markdown("- PDF (.pdf)\n- Word (.docx)")
    st.markdown("---")
    st.markdown("**LLM**\ndeepseek/deepseek-r1 via OpenRouter (>12B)")
 
uploaded_file = st.file_uploader("Upload your document", type=["pdf", "docx"])
 
if uploaded_file:
    st.success(f"File uploaded: {uploaded_file.name} — {uploaded_file.size / 1024:.1f} KB")
 
    if st.button("Extract and Analyse", use_container_width=True, type="primary"):
 
        with st.spinner("Extracting text from document..."):
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
            st.error("Could not extract readable text from the document.")
            st.stop()
 
        with st.expander("Raw Extracted Text", expanded=False):
            st.text_area("", raw_text, height=200, label_visibility="collapsed")
 
        detection_reasoning = ""
        if auto_detect:
            with st.spinner("Detecting document type..."):
                detected_type, detection_reasoning = detect_document_type(raw_text)
                doc_type = detected_type
            st.markdown(f'<div class="detected-type">Auto-detected type: <strong>{doc_type}</strong></div>', unsafe_allow_html=True)
            if detection_reasoning:
                st.caption(f"Reason: {detection_reasoning}")
 
        with st.spinner("Generating AI summary..."):
            summary = generate_summary(raw_text, doc_type)
 
        st.markdown("---")
        st.markdown('<div class="section-title">Document Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
 
        with st.spinner("Extracting structured fields..."):
            result = extract_structured_data(raw_text, doc_type)
 
        if "error" in result:
            st.error(f"LLM extraction failed: {result['error']}")
            st.stop()
 
        flat_result = {k: (v.get("value") if isinstance(v, dict) else v) for k, v in result.items()}
        validation_issues = validate_document_data(flat_result, doc_type)
        review_status = get_review_status(validation_issues)
 
        with st.spinner("Running AI quality review..."):
            field_analysis = analyze_fields(result, doc_type, raw_text)
 
        status_classes = {"Auto Approved": "status-approved", "Needs Review": "status-review", "Rejected": "status-rejected"}
        css_class = status_classes.get(review_status, "status-review")
 
        st.markdown("---")
        st.markdown("### Review Status")
        st.markdown(f'<span class="{css_class}">{review_status}</span>', unsafe_allow_html=True)
        st.markdown("")
 
        if validation_issues:
            with st.expander("Validation Issues", expanded=(review_status == "Rejected")):
                for issue in validation_issues:
                    st.warning(f"- {issue}")
        else:
            st.success("All validation checks passed.")
 
        st.markdown("---")
        st.markdown('<div class="section-title">AI Quality Review</div>', unsafe_allow_html=True)
 
        col1, col2 = st.columns(2)
        with col1:
            missing = field_analysis.get("missing_fields", [])
            st.markdown(f"**Missing Fields ({len(missing)})**")
            if missing:
                for f in missing:
                    st.error(f"- {f.replace('_', ' ').title()}")
            else:
                st.success("No missing fields detected.")
 
        with col2:
            unclear = field_analysis.get("unclear_fields", [])
            st.markdown(f"**Unclear / Possibly Incorrect ({len(unclear)})**")
            if unclear:
                for f in unclear:
                    st.warning(f"- {f.replace('_', ' ').title()}")
            else:
                st.success("No unclear fields detected.")
 
        suggestions = field_analysis.get("suggestions", [])
        if suggestions:
            st.markdown("**Suggestions**")
            for s in suggestions:
                st.info(f"- {s}")
 
        st.markdown("---")
        st.markdown('<div class="section-title">Extracted Structured Data</div>', unsafe_allow_html=True)
 
        tab1, tab2 = st.tabs(["Field View", "Raw JSON"])
 
        with tab1:
            confidence_colors = {"High": "confidence-high", "Medium": "confidence-medium", "Low": "confidence-low"}
 
            for key, field_data in result.items():
                if isinstance(field_data, dict):
                    value = field_data.get("value")
                    confidence = field_data.get("confidence", "Low")
                    evidence = field_data.get("evidence")
                else:
                    value = field_data
                    confidence = "Low"
                    evidence = None
 
                col1, col2, col3 = st.columns([2, 2, 1])
                col1.markdown(f"**{key.replace('_', ' ').title()}**")
 
                if isinstance(value, (dict, list)):
                    col2.json(value)
                else:
                    col2.markdown(str(value) if value is not None else "_Not found_")
                    if evidence:
                        col2.markdown(f'<span class="evidence-text">"{evidence}"</span>', unsafe_allow_html=True)
 
                conf_class = confidence_colors.get(confidence, "confidence-low")
                col3.markdown(f'<span class="{conf_class}">{confidence}</span>', unsafe_allow_html=True)
 
                st.divider()
 
        with tab2:
            final_output = {
                "document_type": doc_type,
                "auto_detected": auto_detect,
                "detection_reasoning": detection_reasoning,
                "filename": uploaded_file.name,
                "extracted_at": datetime.utcnow().isoformat() + "Z",
                "summary": summary,
                "review_status": review_status,
                "validation_issues": validation_issues,
                "field_analysis": field_analysis,
                "extracted_data": result
            }
            st.json(final_output)
            json_str = json.dumps(final_output, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_extracted.json",
                mime="application/json"
            )
 
else:
    st.info("Upload a document to get started.")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Upload**\nPDF or Word documents accepted.")
    with col2:
        st.markdown("**Extract**\nAI detects type, extracts fields with confidence and evidence.")
    with col3:
        st.markdown("**Review**\nAI identifies missing or unclear fields and gives suggestions.")
