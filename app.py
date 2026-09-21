from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# ไฟล์ที่ต้องอยู่โฟลเดอร์เดียวกับ app.py
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "titanic_tree.joblib"
SCALER_PATH = BASE_DIR / "scaler.joblib"  # ไม่บังคับ: ใช้เมื่อตอนเทรนมีการ scale ข้อมูล

# คอลัมน์ที่ถูก scale ตอนเทรน (ใช้เฉพาะเมื่อมี scaler.joblib และ scaler ไม่ได้จำชื่อคอลัมน์ไว้เอง)
# ต้องเรียงตามลำดับเดียวกับตอน scaler.fit(...) ใน Colab
SCALED_COLUMNS = ["Age", "Fare"]

# FamilySize = SibSp + Parch + 1 (นับตัวเองด้วย) — ถ้าตอนเทรนไม่ได้ +1 ให้เปลี่ยนเป็น False
FAMILY_INCLUDES_SELF = True

THRESHOLD = 0.5  # ถ้าความน่าจะเป็น >= ค่านี้ ถือว่า "รอดชีวิต"

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_scaler():
    if not SCALER_PATH.exists():
        return None
    return joblib.load(SCALER_PATH)


st.title("🚢 ทำนายการรอดชีวิตบนเรือ Titanic")
st.caption("โมเดล Decision Tree (scikit-learn) — กรอกข้อมูลผู้โดยสารแล้วกดทำนาย")

# ---- โหลดโมเดล ----
if not MODEL_PATH.exists():
    st.error(f"ไม่พบไฟล์ {MODEL_PATH.name} ในโฟลเดอร์ {BASE_DIR}")
    st.stop()

try:
    model = load_model()
    scaler = load_scaler()
except Exception as e:
    st.error(f"โหลดไฟล์ไม่สำเร็จ: {e}")
    st.stop()

# ลำดับคอลัมน์ที่โมเดลต้องการ (อ่านจากโมเดลเอง)
FEATURE_ORDER = list(model.feature_names_in_)

with st.sidebar:
    st.subheader("ข้อมูลโมเดล")
    st.write("ฟีเจอร์: " + ", ".join(FEATURE_ORDER))
    st.write("ความลึกของต้นไม้: " + str(model.get_depth()))
    st.write("Scaler: " + ("✅ ใช้ scaler.joblib" if scaler is not None else "— ไม่ได้ใช้"))

# ---- ฟอร์มกรอกข้อมูล ----
with st.form("predict_form"):
    pclass = st.selectbox("ชั้นโดยสาร (Pclass)", [1, 2, 3], index=2)
    sex = st.radio("เพศ", ["ชาย", "หญิง"], horizontal=True)
    age = st.number_input("อายุ (ปี)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    fare = st.number_input("ค่าโดยสาร (Fare)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
    sibsp = st.number_input("จำนวนพี่น้อง/คู่สมรสที่มาด้วย (SibSp)", min_value=0, max_value=10, value=0, step=1)
    parch = st.number_input("จำนวนพ่อแม่/ลูกที่มาด้วย (Parch)", min_value=0, max_value=10, value=0, step=1)
    submitted = st.form_submit_button("ทำนาย", type="primary")

# ---- ทำนาย ----
if submitted:
    family_size = int(sibsp) + int(parch) + (1 if FAMILY_INCLUDES_SELF else 0)
    raw = {
        "Pclass": pclass,
        "Sex_female": 1 if sex == "หญิง" else 0,
        "Age": age,
        "Fare": fare,
        "FamilySize": family_size,
    }
    X_raw = pd.DataFrame([raw])[FEATURE_ORDER]  # เรียงคอลัมน์ให้ตรงกับตอนเทรน
    X_model = X_raw.astype(float)

    if scaler is not None:
        cols = list(getattr(scaler, "feature_names_in_", SCALED_COLUMNS))
        try:
            X_model[cols] = scaler.transform(X_raw[cols].astype(float))
        except Exception as e:
            st.error(f"ใช้ scaler ไม่สำเร็จ: {e}\n\nตรวจรายการ SCALED_COLUMNS ใน app.py ให้ตรงกับตอน fit")
            st.stop()

    proba = model.predict_proba(X_model)[0]
    prob_survive = float(proba[list(model.classes_).index(1)])

    st.subheader("ผลการทำนาย")
    st.metric("ความน่าจะเป็นที่จะรอดชีวิต", f"{prob_survive:.1%}")
    st.progress(min(max(prob_survive, 0.0), 1.0))

    if prob_survive >= THRESHOLD:
        st.success("โมเดลทำนายว่า: **รอดชีวิต** ✅")
    else:
        st.warning("โมเดลทำนายว่า: **ไม่รอดชีวิต** ❌")

    with st.expander("ดูค่าที่ส่งเข้าโมเดล"):
        st.write("ค่าดิบที่กรอก")
        st.dataframe(X_raw)
        st.write("ค่าที่ส่งเข้าโมเดลจริง (หลัง scale)")
        st.dataframe(X_model)
