import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path


USERS = {
    "natan": "Natan2026",
    "naud": "Naud2026",
    "Delfina": "Delfina2026",
    "Karmen": "Karmen2026"
}

if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state.user = username
            st.rerun()
        else:
            st.error("Wrong credentials")

    st.stop()

# =========================
# SETTINGS
# =========================

FILE = f"ptsi_data_{st.session_state.user}.csv"
WINDOW = 7
EPS = 0.1
K = 3
T = -1.5

# =========================
# LOAD DATA (SAFE)
# =========================

if Path(FILE).exists():
    data = pd.read_csv(FILE)

    if "date" not in data.columns:
        data["date"] = pd.date_range(end=pd.Timestamp.today(), periods=len(data))

    data["date"] = pd.to_datetime(data["date"])

else:
    data = pd.DataFrame(columns=[
        "date","stress","motivation","readiness","HRV","RHR","PTSI"
    ])

# =========================
# FUNCTIONS
# =========================

def clamp(x):
    return max(EPS, min(1, x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_ptsi(df):
    ptsi_list = []

    for i in range(len(df)):
        if i < WINDOW - 1:
            ptsi_list.append(np.nan)
            continue

        recent = df.iloc[i-WINDOW+1:i+1]

        HRV_base = recent["HRV"].median()
        HRV_MAD = np.median(np.abs(recent["HRV"] - HRV_base)) or 1

        RHR_base = recent["RHR"].median()
        RHR_MAD = np.median(np.abs(recent["RHR"] - RHR_base)) or 1

        row = df.iloc[i]

        stress_n = clamp(row["stress"] / 10)
        mot_n = clamp(row["motivation"] / 10)
        read_n = clamp(row["readiness"] / 10)

        z_hrv = (HRV_base - row["HRV"]) / (1.4826 * HRV_MAD)
        hrv_n = clamp(sigmoid(0.7 * z_hrv))

        z_rhr = (row["RHR"] - RHR_base) / (1.4826 * RHR_MAD)
        rhr_n = clamp(sigmoid(0.7 * z_rhr))

        Z = (
            1.2 * np.log(stress_n)
            - 0.8 * np.log(mot_n)
            - 0.8 * np.log(read_n)
            + 1.0 * np.log(hrv_n)
            + 1.0 * np.log(rhr_n)
        )

        PTSI = sigmoid(K * (Z - T))
        ptsi_list.append(PTSI)

    return ptsi_list

def ai_feedback(row, df):

    stress = row["stress"]
    motivation = row["motivation"]
    readiness = row["readiness"]
    HRV = row["HRV"]
    RHR = row["RHR"]
    ptsi = row["PTSI"]

    recent = df.tail(7)

    HRV_base = recent["HRV"].median()
    RHR_base = recent["RHR"].median()

    # =========================
    # STATES
    # =========================
    phys_fatigue = HRV < HRV_base * 0.9 and RHR > RHR_base * 1.05
    psych_fatigue = stress > 7 and motivation < 4
    mixed_state = phys_fatigue and psych_fatigue

    high_ready = HRV > HRV_base * 1.05 and RHR < RHR_base * 0.95 and motivation > 6

    # =========================
    # TREND
    # =========================
    trend = "stable"

    if len(df) >= 3:
        last3 = df.tail(3)["PTSI"]
        if last3.isna().sum() == 0:
            if last3.iloc[-1] > last3.iloc[0] + 0.1:
                trend = "fatigue increasing"
            elif last3.iloc[-1] < last3.iloc[0] - 0.1:
                trend = "recovering"

    # =========================
    # MOMENTUM (5 days)
    # =========================
    fatigue_accumulation = False
    if len(df) >= 5:
        last5 = df.tail(5)["PTSI"]
        if last5.isna().sum() == 0 and last5.is_monotonic_increasing:
            fatigue_accumulation = True

    # =========================
    # HRV suppression
    # =========================
    hrv_suppression = False
    if len(df) >= 3:
        last3_hrv = df.tail(3)["HRV"]
        if (last3_hrv < HRV_base * 0.92).all():
            hrv_suppression = True

    # =========================
    # VARIABILITY
    # =========================
    variability = recent["PTSI"].std() if len(recent) > 1 else 0

    # =========================
    # CONFLICT
    # =========================
    conflict = None
    if readiness > 7 and phys_fatigue:
        conflict = "Feels good but physiology shows fatigue"
    elif readiness < 4 and not phys_fatigue:
        conflict = "Feels bad but physiology is normal"

    # =========================
    # STATE
    # =========================
    if mixed_state:
        state = "Systemic fatigue"
    elif phys_fatigue:
        state = "Physiological fatigue"
    elif psych_fatigue:
        state = "Psychological fatigue"
    elif high_ready:
        state = "High performance"
    else:
        state = "Mixed / compensated"

    # =========================
    # DRIVERS
    # =========================
    drivers = []

    if stress > 7:
        drivers.append("high stress")
    if motivation < 4:
        drivers.append("low motivation")
    if readiness < 5:
        drivers.append("low readiness")
    if HRV < HRV_base * 0.95:
        drivers.append("low HRV")
    if RHR > RHR_base * 1.05:
        drivers.append("high RHR")

    if len(drivers) == 0:
        drivers.append("no clear issues")

    # =========================
    # RISK
    # =========================
    if ptsi > 0.75:
        risk = "High risk"
    elif ptsi > 0.6:
        risk = "Moderate risk"
    elif ptsi > 0.4:
        risk = "Low–moderate"
    else:
        risk = "Low risk"

    # =========================
    # TRAINING
    # =========================
    if ptsi < 0.2:
        training = "High intensity possible"
    elif ptsi < 0.5:
        training = "Normal training"
    elif ptsi < 0.7:
        training = "Reduce intensity"
    else:
        training = "Recovery"

    # =========================
    # INSIGHTS + TIPS
    # =========================
    insights = []
    tips = []

    if fatigue_accumulation:
        insights.append("Fatigue is accumulating over multiple days")
        tips.append("Consider a recovery day")

    if hrv_suppression:
        insights.append("HRV suppressed for several days")
        tips.append("Prioritize sleep and recovery")

    if HRV < HRV_base * 0.92:
        insights.append("HRV is below your normal")
        tips.append("Keep intensity low")

    if RHR > RHR_base * 1.08:
        insights.append("Resting HR is elevated")
        tips.append("Hydrate well and avoid intensity")

    if stress > 7:
        insights.append("Stress levels are high")
        tips.append("Keep training simple and controlled")

    if motivation < 4:
        insights.append("Motivation is low")
        tips.append("Start easy and focus on enjoyment")

    if variability > 0.2:
        insights.append("Your system is unstable")
        tips.append("Avoid large intensity swings")

    if readiness > 80:
        insights.append("You are in a strong performance window")
        tips.append("Good day for a key session")

    if readiness < 40:
        insights.append("You are not well recovered")
        tips.append("Focus on recovery")

    if phys_fatigue:
        insights.append("Fatigue is mainly physiological")

    if psych_fatigue:
        insights.append("Fatigue is mainly psychological")

    # =========================
    # OUTPUT
    # =========================
    text = f"""
🧠 STATE: {state}

📉 TREND: {trend}

🔍 DRIVERS: {", ".join(drivers)}

⚠️ RISK: {risk}

🏋️ TRAINING: {training}

🧠 INSIGHTS:
- {"\n- ".join(insights) if insights else "No major insights"}

💡 WHAT TO DO:
- {"\n- ".join(tips) if tips else "Train as planned"}
"""

    if conflict:
        text += f"\n\n⚖️ CONFLICT: {conflict}"

    return text

st.sidebar.write(f"Logged in as: {st.session_state.user}")

if st.sidebar.button("Logout"):
    st.session_state.user = None
    st.rerun()

st.title("PTSI – Recovery & Readiness")

st.subheader("➕ Add Day")

date = st.date_input("Date")

col1, col2 = st.columns(2)

with col1:
    stress = st.slider("Stress", 0, 10, 5)
    motivation = st.slider("Motivation", 0, 10, 5)
    readiness = st.slider("Readiness", 0, 10, 5)

with col2:
    HRV = st.number_input("HRV", value=60)
    RHR = st.number_input("Resting HR", value=55)

colA, colB, colC = st.columns(3)

with colA:
    if st.button("💾 Save Day"):
        new_row = pd.DataFrame([{
            "date": pd.to_datetime(date),
            "stress": stress,
            "motivation": motivation,
            "readiness": readiness,
            "HRV": HRV,
            "RHR": RHR
        }])

        data = pd.concat([data, new_row], ignore_index=True)
        data["date"] = pd.to_datetime(data["date"])
        data = data.sort_values("date")

        data["PTSI"] = calculate_ptsi(data)
        data.to_csv(FILE, index=False)

        st.success("Saved!")

with colB:
    if st.button("🗑️ Clear All Data"):
        data = pd.DataFrame(columns=[
            "date","stress","motivation","readiness","HRV","RHR","PTSI"
        ])
        data.to_csv(FILE, index=False)
        st.success("All data cleared!")

with colC:
    if st.button("❌ Delete Last Day"):
        if len(data) > 0:
            data = data.iloc[:-1]
            data["PTSI"] = calculate_ptsi(data)
            data.to_csv(FILE, index=False)
            st.success("Last day removed")

st.subheader("✏️ Edit Data")

edited_data = st.data_editor(data, num_rows="dynamic", use_container_width=True)

if st.button("💾 Save Changes"):
    edited_data["date"] = pd.to_datetime(edited_data["date"])
    edited_data = edited_data.sort_values("date")
    edited_data["PTSI"] = calculate_ptsi(edited_data)
    edited_data.to_csv(FILE, index=False)
    st.success("Changes saved!")

st.subheader("📊 Data")
st.dataframe(data)

if len(data) >= WINDOW:
    st.subheader("📈 Readiness Trend (%)")
    data["READINESS"] = (1 - data["PTSI"]) * 100
    st.line_chart(data.set_index("date")["READINESS"])

# =========================
# TODAY
# =========================

if len(data) > 0:
    last = data.iloc[-1]

    if not np.isnan(last["PTSI"]):
        ptsi = last["PTSI"]
        readiness_pct = (1 - ptsi) * 100

        st.subheader("Today")
        st.metric("Readiness", f"{readiness_pct:.0f}%")

        if readiness_pct >= 75:
            st.success("🟢 High readiness – push")
        elif readiness_pct >= 60:
            st.info("🟡 Good – normal")
        elif readiness_pct >= 40:
            st.warning("🟠 Be careful")
        else:
            st.error("🔴 Recovery")

        st.subheader("Reccomandations")
        st.info(ai_feedback(last, data))

    else:
        st.warning("Need at least 7 days baseline")
