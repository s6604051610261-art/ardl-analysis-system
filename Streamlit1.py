# =========================
# IMPORT
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from google import genai

# =========================
# PAGE SETUP
# =========================
st.set_page_config(
    page_title="ARDL + ECM Analysis",
    layout="wide"
)

st.title("📊 ARDL + ECM + Gemini AI System")

# =========================
# GEMINI API
# =========================
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

# =========================
# UPLOAD FILE
# =========================
uploaded_file = st.file_uploader(
    "📂 Upload Excel File",
    type=["xlsx"]
)

if uploaded_file is None:
    st.info("กรุณาอัปโหลดไฟล์ Excel ก่อน")
    st.stop()

# =========================
# READ DATA
# =========================
try:
    df = pd.read_excel(uploaded_file)

except Exception as e:
    st.error(f"ไม่สามารถอ่านไฟล์ได้\n\n{e}")
    st.stop()

# =========================
# REQUIRED COLUMNS
# =========================
required_columns = [
    "GDP",
    "PIT",
    "PGDP",
    "MLR",
    "CPIH",
    "Wealth",
    "RCP"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    st.error(f"ไม่พบคอลัมน์ : {missing}")
    st.stop()

# =========================
# DATASET PREVIEW
# =========================
st.success("✅ Upload สำเร็จ")

col1, col2 = st.columns(2)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

st.subheader("📋 Dataset Information")

info = pd.DataFrame({
    "Column": df.columns,
    "Type": df.dtypes.astype(str)
})

st.dataframe(info)

# =========================
# RUN BUTTON
# =========================
if st.button("🚀 Run ARDL Analysis"):

    with st.spinner("Running Analysis..."):

        # =====================
        # CREATE VARIABLES
        # =====================
        df["RDI"] = ((df["GDP"] - df["PIT"]) / df["PGDP"]) * 100

        df["INFE"] = (
            (df["CPIH"] - df["CPIH"].shift(4))
            / df["CPIH"].shift(4)
        ) * 100

        df["RIR"] = df["MLR"] - df["INFE"]

        df["ln_RCP"] = np.log(df["RCP"].replace(0, np.nan))
        df["ln_RDI"] = np.log(df["RDI"].replace(0, np.nan))
        df["ln_Wealth"] = np.log(df["Wealth"].replace(0, np.nan))

        df_long = df.dropna().copy()

        # =====================
        # LONG RUN
        # =====================
        X_long = sm.add_constant(
            df_long[
                [
                    "ln_RDI",
                    "ln_Wealth",
                    "RIR"
                ]
            ]
        )

        y_long = df_long["ln_RCP"]

        longrun = sm.OLS(
            y_long,
            X_long
        ).fit()

        df_long["ecm"] = longrun.resid

        longrun_df = pd.DataFrame({
            "Variable": longrun.params.index,
            "Coefficient": longrun.params.values,
            "t-stat": longrun.tvalues.values,
            "P-value": longrun.pvalues.values
        })

        st.subheader("📈 Long Run Results")
        st.dataframe(longrun_df)

        # =====================
        # SHORT RUN
        # =====================
        df_long["dln_RCP"] = df_long["ln_RCP"].diff()

        df_long["dln_RDI"] = df_long["ln_RDI"].diff()

        df_long["ecm_l1"] = df_long["ecm"].shift(1)

        df_short = df_long.dropna().copy()

        X_short = sm.add_constant(
            df_short[
                [
                    "dln_RDI",
                    "ecm_l1"
                ]
            ]
        )

        y_short = df_short["dln_RCP"]

        short_run = sm.OLS(
            y_short,
            X_short
        ).fit()

        shortrun_df = pd.DataFrame({
            "Variable": short_run.params.index,
            "Coefficient": short_run.params.values,
            "t-stat": short_run.tvalues.values,
            "P-value": short_run.pvalues.values
        })

        st.subheader("📉 Short Run Results")
        st.dataframe(shortrun_df)

        # =====================
        # GEMINI
        # =====================
        prompt = f"""
คุณคือผู้เชี่ยวชาญด้านเศรษฐมิติ

Long Run Results

{longrun_df.to_string(index=False)}

Short Run Results

{shortrun_df.to_string(index=False)}

กรุณาอธิบาย

1. ผลระยะยาว
2. ผลระยะสั้น
3. ความหมายของ ECM
4. ข้อเสนอแนะเชิงเศรษฐศาสตร์
"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        st.subheader("🤖 Gemini AI Interpretation")
        st.write(response.text)

        # =====================
        # PDF
        # =====================
        pdf_path = "ARDL_Report.pdf"

        doc = SimpleDocTemplate(pdf_path)

        styles = getSampleStyleSheet()

        style = ParagraphStyle(
            "normal",
            parent=styles["BodyText"],
            fontSize=11
        )

        elements = []

        elements.append(
            Paragraph("<b>ARDL REPORT</b>", style)
        )

        elements.append(
            Spacer(1, 20)
        )

        elements.append(
            Paragraph(
                response.text.replace("\n", "<br/>"),
                style
            )
        )

        doc.build(elements)

        with open(pdf_path, "rb") as f:

            st.download_button(
                "📄 Download PDF Report",
                data=f,
                file_name="ARDL_Report.pdf",
                mime="application/pdf"
            )

    st.success("✅ วิเคราะห์เสร็จเรียบร้อย")