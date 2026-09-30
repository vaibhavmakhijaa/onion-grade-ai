# ============================================================
# ONION GRADE AI
# Smart India Hackathon 2026
# Team: Runtime Terror
# Team Leader: Vaibhav Makhija
# ============================================================

import streamlit as st
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import numpy as np
import joblib
from fpdf import FPDF
from datetime import datetime


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Onion Grade AI",
    page_icon="🧅",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.html("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(93, 135, 80, 0.12),
            transparent 30%
        ),
        radial-gradient(
            circle at 95% 5%,
            rgba(214, 170, 70, 0.08),
            transparent 25%
        ),
        #0b100d;
}

.main .block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {
    background: #0d130f;
    border-right: 1px solid #29332c;
}


/* ================= BRAND ================= */

.brand {
    display: flex;
    align-items: center;
    gap: 15px;
    margin-bottom: 10px;
}

.brand-icon {
    width: 60px;
    height: 60px;
    border-radius: 17px;
    background: linear-gradient(
        135deg,
        #e2b653,
        #a66e28
    );

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 32px;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.35);
}

.brand-title {
    color: #f4f7f4;
    font-size: 2.45rem;
    font-weight: 850;
    letter-spacing: -1px;
}

.brand-subtitle {
    color: #92a096;
    font-size: 0.92rem;
    margin-top: 5px;
}


/* ================= TEAM BADGE ================= */

.team-badge {
    display: inline-block;

    margin-top: 12px;

    padding: 7px 14px;

    border-radius: 999px;

    background: rgba(226,182,83,0.10);

    border: 1px solid rgba(226,182,83,0.35);

    color: #e2b653;

    font-size: 0.78rem;

    font-weight: 800;

    letter-spacing: 0.5px;
}


/* ================= HERO ================= */

.hero {
    margin-top: 28px;

    padding: 32px;

    border-radius: 23px;

    border: 1px solid #2c382f;

    background:
        linear-gradient(
            135deg,
            rgba(28,43,33,0.96),
            rgba(14,20,16,0.98)
        );

    box-shadow:
        0 20px 55px rgba(0,0,0,0.20);
}

.hero-title {
    color: #f2f6f3;

    font-size: 2rem;

    font-weight: 800;

    letter-spacing: -0.5px;

    margin-bottom: 9px;
}

.hero-text {
    color: #a5b1a8;

    font-size: 0.98rem;

    line-height: 1.7;

    max-width: 850px;
}


/* ================= PIPELINE ================= */

.pipeline {
    display: flex;

    align-items: center;

    flex-wrap: wrap;

    gap: 8px;

    margin-top: 23px;
}

.pipeline-step {
    padding: 8px 13px;

    border-radius: 10px;

    background: #111813;

    border: 1px solid #2c382f;

    color: #dce5de;

    font-size: 0.78rem;

    font-weight: 700;
}

.pipeline-arrow {
    color: #c59c4d;

    font-size: 1rem;

    font-weight: 700;
}


/* ================= SECTION HEADINGS ================= */

.section-title {
    margin-top: 32px;

    margin-bottom: 5px;

    color: #edf2ee;

    font-size: 1.3rem;

    font-weight: 800;
}

.section-description {
    color: #7e8b82;

    font-size: 0.86rem;

    margin-bottom: 15px;
}


/* ================= TEAM CARD ================= */

.team-card {
    padding: 18px;

    border-radius: 16px;

    background: #141b16;

    border: 1px solid #29352d;
}

.team-name {
    color: #e2b653;

    font-size: 1.08rem;

    font-weight: 850;

    margin-bottom: 8px;
}

.team-detail {
    color: #9ba79e;

    font-size: 0.81rem;

    margin-top: 5px;
}


/* ================= RESULT CARDS ================= */

.result-card {
    padding: 19px;

    min-height: 118px;

    border-radius: 17px;

    background: #111813;

    border: 1px solid #29352d;

    box-shadow:
        0 8px 25px rgba(0,0,0,0.12);
}

.healthy-card {
    border-top: 3px solid #62b574;
}

.damaged-card {
    border-top: 3px solid #d3a34d;
}

.rotten-card {
    border-top: 3px solid #d45f5f;
}

.sprouted-card {
    border-top: 3px solid #71a9e2;
}

.result-label {
    color: #8b988f;

    font-size: 0.76rem;

    font-weight: 750;

    text-transform: uppercase;

    letter-spacing: 0.8px;
}

.result-number {
    color: #f2f6f3;

    font-size: 2.1rem;

    font-weight: 850;

    margin-top: 6px;
}

.result-sub {
    color: #6f7c73;

    font-size: 0.74rem;

    margin-top: 3px;
}


/* ================= ASSESSMENT ================= */

.assessment {
    padding: 28px;

    border-radius: 20px;

    border: 1px solid #364238;

    background:
        linear-gradient(
            135deg,
            #151e17,
            #101611
        );

    text-align: center;
}

.assessment-title {
    color: #89968d;

    font-size: 0.73rem;

    text-transform: uppercase;

    letter-spacing: 1.4px;

    font-weight: 800;
}

.assessment-grade {
    color: #e2b653;

    font-size: 2.2rem;

    font-weight: 900;

    margin-top: 7px;
}

.assessment-percent {
    color: #cbd5ce;

    margin-top: 5px;

    font-size: 0.88rem;
}


/* ================= INFO BOX ================= */

.info-box {
    padding: 18px 20px;

    border-radius: 15px;

    background: #111813;

    border: 1px solid #29352d;

    color: #9eaaa1;

    font-size: 0.86rem;

    line-height: 1.65;
}


/* ================= EMPTY STATE ================= */

.empty-state {
    margin-top: 38px;

    padding: 50px 30px;

    border-radius: 22px;

    border: 1px dashed #354139;

    background: #0f1511;

    text-align: center;
}

.empty-icon {
    font-size: 3.2rem;

    margin-bottom: 10px;
}

.empty-title {
    color: #edf2ee;

    font-size: 1.3rem;

    font-weight: 800;
}

.empty-text {
    color: #7f8b82;

    margin-top: 7px;

    font-size: 0.87rem;
}


/* ================= BUTTONS ================= */

.stButton > button {
    min-height: 48px;

    border-radius: 12px;

    font-weight: 800;

    letter-spacing: 0.2px;
}


/* ================= FILE UPLOADER ================= */

[data-testid="stFileUploader"] {
    background: #111813;

    border-radius: 15px;

    padding: 7px;
}


/* ================= DATAFRAME ================= */

[data-testid="stDataFrame"] {
    border-radius: 15px;

    overflow: hidden;

    border: 1px solid #29352d;
}


/* ================= FOOTER ================= */

.footer {
    margin-top: 55px;

    padding-top: 20px;

    border-top: 1px solid #252e28;

    text-align: center;

    color: #66736a;

    font-size: 0.75rem;

    line-height: 1.8;
}

</style>
""")


# ============================================================
# 3. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

YOLO_MODEL_PATH = (
    BASE_DIR.parent
    / "runs"
    / "detect"
    / "onion_detector_v3"
    / "weights"
    / "best.pt"
)

CLASSIFIER_PATH = (
    BASE_DIR
    / "models"
    / "onion_classifier.pkl"
)


# ============================================================
# 4. CHECK MODEL FILES
# ============================================================

if not YOLO_MODEL_PATH.exists():

    st.error(
        "YOLO model was not found:\n\n"
        f"{YOLO_MODEL_PATH}"
    )

    st.stop()


if not CLASSIFIER_PATH.exists():

    st.error(
        "Classifier model was not found:\n\n"
        f"{CLASSIFIER_PATH}"
    )

    st.stop()


# ============================================================
# 5. LOAD MODELS
# ============================================================

detector = YOLO(
    str(YOLO_MODEL_PATH)
)

classifier = joblib.load(
    CLASSIFIER_PATH
)


# ============================================================
# 6. FEATURE EXTRACTION
# ============================================================
#
# IMPORTANT:
# These are the SAME 5 features used for training.
#
# 1. Hue
# 2. Saturation
# 3. Value
# 4. Dark pixel ratio
# 5. Green pixel ratio
#
# ============================================================

def extract_features(image):

    if image is None or image.size == 0:

        return None


    image = cv2.resize(
        image,
        (128, 128)
    )


    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )


    # Feature 1
    hue = np.mean(
        hsv[:, :, 0]
    )


    # Feature 2
    saturation = np.mean(
        hsv[:, :, 1]
    )


    # Feature 3
    value = np.mean(
        hsv[:, :, 2]
    )


    # Feature 4
    dark_pixels = np.sum(
        hsv[:, :, 2] < 80
    )


    dark_pixel_ratio = (
        dark_pixels / (128 * 128)
    )


    # Feature 5
    green_mask = (
        (hsv[:, :, 0] > 35) &
        (hsv[:, :, 0] < 90) &
        (hsv[:, :, 1] > 60)
    )


    green_pixel_ratio = (
        np.sum(green_mask)
        / (128 * 128)
    )


    return [
        hue,
        saturation,
        value,
        dark_pixel_ratio,
        green_pixel_ratio
    ]


# ============================================================
# 7. PDF REPORT
# ============================================================

def create_pdf_report(
    total,
    counts,
    healthy_percentage,
    grade,
    size_note
):

    pdf = FPDF()

    pdf.add_page()


    # TITLE
    pdf.set_font(
        "Arial",
        "B",
        20
    )

    pdf.cell(
        0,
        15,
        "ONION GRADE AI",
        ln=True,
        align="C"
    )


    pdf.set_font(
        "Arial",
        "B",
        14
    )

    pdf.cell(
        0,
        10,
        "Digital Quality Assessment Report",
        ln=True,
        align="C"
    )


    pdf.ln(10)


    # PROJECT INFO
    pdf.set_font(
        "Arial",
        "",
        11
    )


    date = datetime.now().strftime(
        "%d-%m-%Y %H:%M"
    )


    pdf.cell(
        0,
        8,
        f"Assessment Date: {date}",
        ln=True
    )


    pdf.cell(
        0,
        8,
        "Team: Runtime Terror",
        ln=True
    )


    pdf.cell(
        0,
        8,
        "Team Leader: Vaibhav Makhija",
        ln=True
    )


    pdf.cell(
        0,
        8,
        "Smart India Hackathon 2026",
        ln=True
    )


    pdf.ln(6)


    # BATCH SUMMARY
    pdf.set_font(
        "Arial",
        "B",
        13
    )


    pdf.cell(
        0,
        10,
        "Batch Summary",
        ln=True
    )


    pdf.set_font(
        "Arial",
        "",
        11
    )


    pdf.cell(
        0,
        8,
        f"Total onions detected: {total}",
        ln=True
    )


    pdf.ln(5)


    # QUALITY BREAKDOWN
    pdf.set_font(
        "Arial",
        "B",
        13
    )


    pdf.cell(
        0,
        10,
        "Quality Breakdown",
        ln=True
    )


    pdf.set_font(
        "Arial",
        "",
        11
    )


    for category, count in counts.items():

        percentage = (
            count / total * 100
            if total > 0
            else 0
        )


        pdf.cell(
            0,
            8,
            f"{category.capitalize()}: "
            f"{count} "
            f"({percentage:.1f}%)",
            ln=True
        )


    pdf.ln(5)


    # ASSESSMENT
    pdf.set_font(
        "Arial",
        "B",
        13
    )


    pdf.cell(
        0,
        10,
        "Prototype Grading Result",
        ln=True
    )


    pdf.set_font(
        "Arial",
        "",
        11
    )


    pdf.cell(
        0,
        8,
        f"Healthy percentage: "
        f"{healthy_percentage:.1f}%",
        ln=True
    )


    pdf.cell(
        0,
        8,
        f"Prototype result: {grade}",
        ln=True
    )


    pdf.ln(5)


    # SIZE
    pdf.set_font(
        "Arial",
        "B",
        13
    )


    pdf.cell(
        0,
        10,
        "Size Assessment",
        ln=True
    )


    pdf.set_font(
        "Arial",
        "",
        11
    )


    pdf.multi_cell(
        0,
        7,
        size_note
    )


    pdf.ln(8)


    # DISCLAIMER
    pdf.set_font(
        "Arial",
        "I",
        9
    )


    pdf.multi_cell(
        0,
        6,
        "Disclaimer: Onion Grade AI is an "
        "AI-assisted prototype. Visual quality "
        "classification and grading are approximate. "
        "Physical size measurement is not calibrated "
        "in this prototype. Official grading standards "
        "and larger real-world datasets should be "
        "incorporated before production use."
    )


    return bytes(
        pdf.output()
    )


# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🧅 Onion Grade AI"
    )

    st.caption(
        "Smart onion quality assessment"
    )


    st.markdown("---")


    st.html("""
    <div class="team-card">

        <div class="team-name">
            ⚡ Runtime Terror
        </div>

        <div class="team-detail">
            Smart India Hackathon 2026
        </div>

        <div class="team-detail">
            👤 Team Leader: Vaibhav Makhija
        </div>

    </div>
    """)


    st.markdown("---")


    st.markdown(
        "### ⚙️ AI Pipeline"
    )


    st.markdown(
        """
        **📷 Input Image**

        ↓

        **🔎 YOLO26**

        Individual onion detection

        ↓

        **🧠 Random Forest**

        Quality classification

        ↓

        **📊 Batch Assessment**

        Quality distribution

        ↓

        **📄 Digital Report**
        """
    )


    st.markdown("---")


    st.caption(
        "Prototype • SIH 2026"
    )


# ============================================================
# 9. MAIN HEADER
# ============================================================

st.html("""
<div class="brand">

    <div class="brand-icon">
        🧅
    </div>

    <div>

        <div class="brand-title">
            Onion Grade AI
        </div>

        <div class="brand-subtitle">
            AI-powered onion quality inspection
        </div>

    </div>

</div>


<div class="team-badge">
    ⚡ RUNTIME TERROR &nbsp;•&nbsp; SIH 2026
</div>
""")


# ============================================================
# 10. HERO
# ============================================================

st.html("""
<div class="hero">

    <div class="hero-title">
        From batch image to digital quality assessment.
    </div>

    <div class="hero-text">
        Onion Grade AI detects individual onions,
        analyzes visible quality conditions and
        converts the results into an instant
        batch-level assessment for faster and
        more consistent inspection.
    </div>


    <div class="pipeline">

        <div class="pipeline-step">
            📷 Capture
        </div>

        <div class="pipeline-arrow">
            →
        </div>

        <div class="pipeline-step">
            🔎 YOLO26
        </div>

        <div class="pipeline-arrow">
            →
        </div>

        <div class="pipeline-step">
            🧅 Detect
        </div>

        <div class="pipeline-arrow">
            →
        </div>

        <div class="pipeline-step">
            🧠 Classify
        </div>

        <div class="pipeline-arrow">
            →
        </div>

        <div class="pipeline-step">
            📊 Assess
        </div>

        <div class="pipeline-arrow">
            →
        </div>

        <div class="pipeline-step">
            📄 Report
        </div>

    </div>

</div>
""")


# ============================================================
# 11. HOW IT WORKS
# ============================================================

with st.expander(
    "ℹ️ How does Onion Grade AI work?"
):

    st.markdown(
        """
        ### 1️⃣ Capture

        Upload a photograph containing an onion batch.

        ### 2️⃣ Detect

        YOLO26 identifies individual onions.

        ### 3️⃣ Crop

        Each detected onion is isolated.

        ### 4️⃣ Extract Features

        OpenCV extracts five visual features:

        - Hue
        - Saturation
        - Brightness
        - Dark pixel ratio
        - Green pixel ratio

        ### 5️⃣ Classify

        Random Forest predicts:

        - Healthy
        - Damaged
        - Rotten
        - Sprouted

        ### 6️⃣ Aggregate

        Individual predictions are converted into
        batch-level statistics.

        ### 7️⃣ Report

        A digital quality assessment report is generated.
        """
    )


# ============================================================
# 12. UPLOAD
# ============================================================

st.html("""
<div class="section-title">
    📷 Batch Inspection
</div>

<div class="section-description">
    Upload a photograph containing visible onions
    for AI-assisted quality assessment.
</div>
""")


uploaded_file = st.file_uploader(
    "Upload onion batch image",

    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ],

    label_visibility="collapsed"
)


# ============================================================
# 13. EMPTY STATE
# ============================================================

if uploaded_file is None:

    st.html("""
    <div class="empty-state">

        <div class="empty-icon">
            🧅
        </div>

        <div class="empty-title">
            Ready for batch inspection
        </div>

        <div class="empty-text">
            Upload an onion batch image above to
            begin AI-assisted quality assessment.
        </div>

    </div>
    """)


# ============================================================
# 14. IMAGE UPLOADED
# ============================================================

else:

    image = Image.open(
        uploaded_file
    )


    left, right = st.columns(
        [2.2, 1],
        gap="large"
    )


    # --------------------------------------------------------
    # IMAGE PREVIEW
    # --------------------------------------------------------

    with left:

        st.html("""
        <div class="section-title">
            🖼️ Input Image
        </div>
        """)


        st.image(
            image,
            use_container_width=True
        )


    # --------------------------------------------------------
    # READY PANEL
    # --------------------------------------------------------

    with right:

        st.html("""
        <div class="section-title">
            🔬 Inspection Ready
        </div>

        <div class="info-box">

            <b style="color:#e4ebe6;">
                AI pipeline ready
            </b>

            <br><br>

            🔎 YOLO26 detection<br>
            🧅 Individual onion cropping<br>
            🧠 Quality classification<br>
            📊 Batch aggregation<br>
            📄 Digital reporting

        </div>
        """)


        st.markdown("")


        analyze = st.button(
            "🔍 START AI INSPECTION",
            type="primary",
            use_container_width=True
        )


    # ========================================================
    # 15. ANALYSIS
    # ========================================================

    if analyze:

        with st.spinner(
            "Running YOLO26 detection and quality analysis..."
        ):


            # ------------------------------------------------
            # SAVE ORIGINAL UPLOAD
            # ------------------------------------------------

            original_suffix = (
                Path(uploaded_file.name).suffix
                or ".jpg"
            )


            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=original_suffix
            ) as temp:

                temp.write(
                    uploaded_file.getbuffer()
                )

                temp_path = temp.name


            # ------------------------------------------------
            # YOLO DETECTION
            # ------------------------------------------------

            results = detector.predict(
                source=temp_path,
                conf=0.50
            )


            result = results[0]


            onion_count = len(
                result.boxes
            )


            # ------------------------------------------------
            # READ IMAGE
            # ------------------------------------------------

            original = cv2.imread(
                temp_path
            )


            if original is None:

                st.error(
                    "Could not read uploaded image."
                )

                st.stop()


            # ------------------------------------------------
            # INITIAL COUNTS
            # ------------------------------------------------

            counts = {
                "healthy": 0,
                "damaged": 0,
                "rotten": 0,
                "sprouted": 0
            }


            predictions = []


            # =================================================
            # CLASSIFY EACH DETECTED ONION
            # =================================================

            for i, box in enumerate(
                result.boxes
            ):


                # Bounding box
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )


                width = x2 - x1

                height = y2 - y1


                # 5% margin
                margin_x = int(
                    width * 0.05
                )

                margin_y = int(
                    height * 0.05
                )


                x1_crop = max(
                    0,
                    x1 + margin_x
                )

                y1_crop = max(
                    0,
                    y1 + margin_y
                )

                x2_crop = min(
                    original.shape[1],
                    x2 - margin_x
                )

                y2_crop = min(
                    original.shape[0],
                    y2 - margin_y
                )


                # Crop onion
                crop = original[
                    y1_crop:y2_crop,
                    x1_crop:x2_crop
                ]


                if crop.size == 0:

                    continue


                # Extract features
                features = extract_features(
                    crop
                )


                if features is None:

                    continue


                # Random Forest
                prediction = classifier.predict(
                    [features]
                )[0]


                probabilities = (
                    classifier.predict_proba(
                        [features]
                    )[0]
                )


                confidence = max(
                    probabilities
                )


                # Count prediction
                counts[
                    prediction
                ] += 1


                # Save result
                predictions.append(
                    {
                        "Onion": i + 1,

                        "Quality":
                            prediction.capitalize(),

                        "Confidence":
                            f"{confidence * 100:.1f}%"
                    }
                )


        # ========================================================
        # 16. RESULTS
        # ========================================================

        st.html("""
        <div class="section-title">
            📊 AI Inspection Results
        </div>
        """)


        st.success(
            f"Analysis complete • "
            f"{onion_count} onions detected 🧅"
        )


        # ========================================================
        # 17. PERCENTAGES
        # ========================================================

        if onion_count > 0:

            healthy_pct = (
                counts["healthy"]
                / onion_count
                * 100
            )

            damaged_pct = (
                counts["damaged"]
                / onion_count
                * 100
            )

            rotten_pct = (
                counts["rotten"]
                / onion_count
                * 100
            )

            sprouted_pct = (
                counts["sprouted"]
                / onion_count
                * 100
            )

        else:

            healthy_pct = 0

            damaged_pct = 0

            rotten_pct = 0

            sprouted_pct = 0


        # ========================================================
        # 18. QUALITY CARDS
        # ========================================================

        c1, c2, c3, c4 = st.columns(
            4,
            gap="medium"
        )


        with c1:

            st.html(
                f"""
                <div class="result-card healthy-card">

                    <div class="result-label">
                        🟢 Healthy
                    </div>

                    <div class="result-number">
                        {counts["healthy"]}
                    </div>

                    <div class="result-sub">
                        {healthy_pct:.1f}% of detected batch
                    </div>

                </div>
                """
            )


        with c2:

            st.html(
                f"""
                <div class="result-card damaged-card">

                    <div class="result-label">
                        🟠 Damaged
                    </div>

                    <div class="result-number">
                        {counts["damaged"]}
                    </div>

                    <div class="result-sub">
                        {damaged_pct:.1f}% of detected batch
                    </div>

                </div>
                """
            )


        with c3:

            st.html(
                f"""
                <div class="result-card rotten-card">

                    <div class="result-label">
                        🔴 Rotten
                    </div>

                    <div class="result-number">
                        {counts["rotten"]}
                    </div>

                    <div class="result-sub">
                        {rotten_pct:.1f}% of detected batch
                    </div>

                </div>
                """
            )


        with c4:

            st.html(
                f"""
                <div class="result-card sprouted-card">

                    <div class="result-label">
                        🌱 Sprouted
                    </div>

                    <div class="result-number">
                        {counts["sprouted"]}
                    </div>

                    <div class="result-sub">
                        {sprouted_pct:.1f}% of detected batch
                    </div>

                </div>
                """
            )


        # ========================================================
        # 19. YOLO DETECTION
        # ========================================================

        st.html("""
        <div class="section-title">
            👁️ Computer Vision Detection
        </div>

        <div class="section-description">
            YOLO26 detection results — individual onions
            identified in the uploaded batch.
        </div>
        """)


        annotated_image = result.plot()


        st.image(
            annotated_image,
            channels="BGR",
            use_container_width=True
        )


        # ========================================================
        # 20. ANALYTICS
        # ========================================================

        if onion_count > 0:

            left, right = st.columns(
                2,
                gap="large"
            )


            # ----------------------------------------------------
            # DISTRIBUTION
            # ----------------------------------------------------

            with left:

                st.html("""
                <div class="section-title">
                    📈 Quality Distribution
                </div>

                <div class="section-description">
                    Distribution of predicted onion
                    quality conditions.
                </div>
                """)


                chart_data = {
                    "Healthy": healthy_pct,
                    "Damaged": damaged_pct,
                    "Rotten": rotten_pct,
                    "Sprouted": sprouted_pct
                }


                st.bar_chart(
                    chart_data
                )


            # ----------------------------------------------------
            # ASSESSMENT
            # ----------------------------------------------------

            with right:


                # Prototype assessment rule
                if (
                    healthy_pct >= 80
                    and counts["rotten"] == 0
                    and counts["sprouted"] == 0
                ):

                    grade = "GRADE A"

                else:

                    grade = "URS / LOWER GRADE"


                st.html(
                    f"""
                    <div class="section-title">
                        🏆 Batch Assessment
                    </div>

                    <div class="assessment">

                        <div class="assessment-title">
                            Prototype Quality Assessment
                        </div>

                        <div class="assessment-grade">
                            {grade}
                        </div>

                        <div class="assessment-percent">
                            {healthy_pct:.1f}% visually healthy
                        </div>

                    </div>
                    """
                )


        # ========================================================
        # 21. INDIVIDUAL RESULTS
        # ========================================================

        st.html("""
        <div class="section-title">
            🔎 Individual Onion Inspection
        </div>

        <div class="section-description">
            Per-onion predictions generated by the
            quality classification model.
        </div>
        """)


        if predictions:

            st.dataframe(
                predictions,

                use_container_width=True,

                hide_index=True,

                column_config={

                    "Onion":
                        st.column_config.NumberColumn(
                            "Onion #",
                            format="%d"
                        ),

                    "Quality":
                        st.column_config.TextColumn(
                            "Quality"
                        ),

                    "Confidence":
                        st.column_config.TextColumn(
                            "Model Confidence"
                        )
                }
            )


        # ========================================================
        # 22. SIZE ASSESSMENT
        # ========================================================

        st.html("""
        <div class="section-title">
            📏 Size Assessment
        </div>
        """)


        size_note = (
            "Physical onion size is not calibrated in the "
            "current prototype. A future version can use "
            "a reference object or calibrated camera setup "
            "to estimate onion diameter and identify "
            "undersized onions."
        )


        st.html(
            f"""
            <div class="info-box">

                <b style="color:#e4ebe6;">
                    Prototype limitation
                </b>

                <br><br>

                {size_note}

            </div>
            """
        )


        # ========================================================
        # 23. PDF REPORT
        # ========================================================

        st.html("""
        <div class="section-title">
            📄 Digital Quality Report
        </div>

        <div class="section-description">
            Generate a portable assessment summary
            for digital procurement records.
        </div>
        """)


        pdf_data = create_pdf_report(
            total=onion_count,

            counts=counts,

            healthy_percentage=healthy_pct,

            grade=grade,

            size_note=size_note
        )


        st.download_button(
            label="📄 GENERATE & DOWNLOAD QUALITY REPORT",

            data=pdf_data,

            file_name="onion_quality_report.pdf",

            mime="application/pdf",

            use_container_width=True
        )


        # ========================================================
        # 24. TECHNICAL DETAILS
        # ========================================================

        with st.expander(
            "🧠 Technical Details"
        ):

            st.markdown(
                """
                ### 🔎 Object Detection

                **YOLO26**

                Detects individual onions from the batch image.

                ### 🖼️ Image Processing

                **OpenCV**

                Extracts five numerical visual features:

                - Hue
                - Saturation
                - Brightness / Value
                - Dark pixel ratio
                - Green pixel ratio

                ### 🧠 Machine Learning

                **Random Forest**

                Classifies each detected onion as:

                - Healthy
                - Damaged
                - Rotten
                - Sprouted

                ### 📊 Batch Analysis

                Individual predictions are aggregated into
                batch-level quality statistics.

                ### 📄 Digital Reporting

                Results can be exported as a PDF report.
                """
            )


        # ========================================================
        # 25. PROTOTYPE STATUS
        # ========================================================

        st.html("""
        <div class="info-box"
             style="margin-top:25px;">

            <b style="color:#e4ebe6;">
                Prototype Status
            </b>

            <br><br>

            Onion Grade AI is an AI-assisted prototype.
            Visual classification and grading are approximate
            and require validation using larger real-world
            datasets and applicable official grading standards.

        </div>
        """)


# ============================================================
# 26. FOOTER
# ============================================================

st.html("""
<div class="footer">

    <b style="color:#aab5ad;">
        Onion Grade AI
    </b>

    &nbsp; • &nbsp;

    Built by

    <b style="color:#aab5ad;">
        Runtime Terror
    </b>

    &nbsp; • &nbsp;

    Smart India Hackathon 2026

    <br>

    Team Leader:

    <b style="color:#aab5ad;">
        Vaibhav Makhija
    </b>

</div>
""")