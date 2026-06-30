
#ลองเอาไปแสดงผลในเว็ปไซต์
import streamlit as st

st.set_page_config(page_title="ARDL Analysis", layout="wide")

st.title("📊ARDL Analysis System")

st.write("ระบบวิเคราะห์ ARDL และสร้างรายงานอัตโนมัติ")

uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx"]
)

if uploaded_file is not None:
    st.success("Upload สำเร็จ")
    
import pandas as pd
if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)

    st.success("Upload สำเร็จ")

    st.subheader("📄 Dataset Preview")

    st.dataframe(df.head())

st.subheader("📊 Dataset Information")

st.write(f"จำนวนข้อมูล : {df.shape[0]} แถว")
st.write(f"จำนวนตัวแปร : {df.shape[1]} ตัว")

if st.button("🚀 Start Analysis"):
    st.write("กำลังวิเคราะห์...")


# =====================================
# 1. IMPORT LIBRARIES
# =====================================
import numpy as np
import pandas as pd
import statsmodels.api as sm
import ollama
import os

from reportlab.platypus import PageBreak, SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from xml.sax.saxutils import escape


# =====================================
# 2. LOAD DATA
# =====================================
file_path = r"C:\Users\HP\Downloads\NXPO\ARDL_ECM_Sample_Data.xlsx"
df = pd.read_excel(file_path)

print("Data Loaded:", df.shape)
print(df.head())


# =====================================
# 3. CREATE VARIABLES
# =====================================

# Real Disposable Income
df["RDI"] = ((df["GDP"] - df["PIT"]) / df["PGDP"]) * 100

# Inflation (YoY 4 quarters)
df["INFE"] = ((df["CPIH"] - df["CPIH"].shift(4)) / df["CPIH"].shift(4)) * 100

# Real Interest Rate
df["RIR"] = df["MLR"] - df["INFE"]


# =====================================
# 4. LOG TRANSFORM (SAFE)
# =====================================
df["ln_RCP"] = np.log(df["RCP"].replace(0, np.nan))
df["ln_RDI"] = np.log(df["RDI"].replace(0, np.nan))
df["ln_Wealth"] = np.log(df["Wealth"].replace(0, np.nan))


# =====================================
# 5. LONG-RUN MODEL (OLS)
# =====================================
df_long = df.dropna().copy()

X_long = sm.add_constant(df_long[["ln_RDI", "ln_Wealth", "RIR"]])
y_long = df_long["ln_RCP"]

longrun = sm.OLS(y_long, X_long).fit()
# ===== ADD HERE =====
longrun_df = pd.DataFrame({
    "Variable": longrun.params.index,
    "Coefficient": longrun.params.values,
    "t-stat": longrun.tvalues.values,
    "P-value": longrun.pvalues.values
})
print("\n================ LONG RUN ================\n")
print(longrun.summary())


# Residuals (ECM term)
df_long["ecm"] = longrun.resid


# =====================================
# 6. SHORT-RUN MODEL (ECM)
# =====================================
df_long["dln_RCP"] = df_long["ln_RCP"].diff()
df_long["dln_RDI"] = df_long["ln_RDI"].diff()

df_long["dln_Wealth_l3"] = df_long["ln_Wealth"].shift(3).diff()
df_long["dRIR_l1"] = df_long["RIR"].shift(1).diff()

df_long["ecm_l1"] = df_long["ecm"].shift(1)

df_short = df_long.dropna().copy()

X_short = sm.add_constant(df_short[
    ["dln_RDI", "dln_Wealth_l3", "dRIR_l1", "ecm_l1"]
])

y_short = df_short["dln_RCP"]

shortrun = sm.OLS(y_short, X_short).fit(cov_type="HC0")

# ===== ADD HERE =====
shortrun_df = pd.DataFrame({
    "Variable": shortrun.params.index,
    "Coefficient": shortrun.params.values,
    "t-stat": shortrun.tvalues.values,
    "P-value": shortrun.pvalues.values
})

print("\n================ SHORT RUN ================\n")
print(shortrun.summary())


# =====================================
# 7. EXTRACT COEFFICIENTS
# =====================================
longrun_coef = longrun.params.to_string()
shortrun_coef = shortrun.params.to_string()


# =====================================
# 8. AI REPORT (OLLAMA)
# =====================================
prompt = f"""
คุณคือผู้เชี่ยวชาญเศรษฐมิติ

ผล Long-run:
{longrun_coef}

ผล Short-run:
{shortrun_coef}

กรุณาอธิบาย:
1. ผลระยะยาว
2. ผลระยะสั้น
3. ECM meaning
4. สรุปเศรษฐศาสตร์
5. เขียนเป็นรายงานวิชาการภาษาไทย
"""

response = ollama.chat(
    model="qwen3:4b",
    messages=[{"role": "user", "content": prompt}]
)

report_text = response["message"]["content"]

print("\nAI REPORT GENERATED\n")


# =====================================
# 9. SAVE TXT (CLEAN)
# =====================================
txt_path = "ARDL_Report.txt"

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(report_text)

print("TXT saved:", txt_path)
# helper function (ต้องอยู่ก่อนใช้)
# =====================================
from reportlab.lib import colors  # ต้องอยู่บนสุดของไฟล์ หรือก่อนใช้ฟังก์ชัน

def dataframe_to_table(df):
    data = [df.columns.tolist()] + df.values.tolist()

    table = Table(data)

    table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),

        ('FONTNAME', (0,0), (-1,-1), 'THSarabun'),
        ('FONTSIZE', (0,0), (-1,-1), 12),

        ('ALIGN', (0,0), (-1,-1), 'CENTER'),

        # 🔥 สีหัวตาราง
        ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),

        # 🔥 สีพื้นทั้งตาราง
        ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
    ]))

    return table
# =====================================
# 10. CREATE PDF (SAFE THAI FONT)
# =====================================

pdfmetrics.registerFont(
    TTFont("THSarabun", r"C:\Users\HP\Downloads\NXPO\THSarabunNew.ttf")
)

doc = SimpleDocTemplate("ARDL_Report.pdf")

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "ThaiTitle",
    parent=styles["Title"],
    fontName="THSarabun",
    fontSize=20,
    leading=26
)

body_style = ParagraphStyle(
    "ThaiBody",
    parent=styles["BodyText"],
    fontName="THSarabun",
    fontSize=14,
    leading=20
)

elements = []

elements.append(Paragraph("รายงาน ARDL & ECM", title_style))
elements.append(Spacer(1, 12))
# =========================
# 🔥 ใส่ตารางตรงนี้
# =========================

elements.append(Paragraph("ตารางผล Long-run Model", body_style))
elements.append(Spacer(1, 6))
elements.append(dataframe_to_table(longrun_df)) # type: ignore
elements.append(Spacer(1, 12))

elements.append(Paragraph("ตารางผล Short-run ECM Model", body_style))
elements.append(Spacer(1, 6))
elements.append(dataframe_to_table(shortrun_df)) # type: ignore
elements.append(Spacer(1, 12))

elements.append(PageBreak())

safe_text = escape(report_text).replace("\n", "<br/>")

elements.append(Paragraph(safe_text, body_style))

doc.build(elements)

print("PDF created: ARDL_Report.pdf")


# =====================================
# 11. OPEN PDF (WINDOWS ONLY)
# =====================================
pdf_path = os.path.abspath("ARDL_Report.pdf")
os.startfile(pdf_path)

print("DONE - ALL PROCESS COMPLETED")