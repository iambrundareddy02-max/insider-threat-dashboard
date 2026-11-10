"""
Insider Threat Detection & Mathematical Analysis Engine
Ported and enhanced from original main.py.
Preserves all mathematical calculations:
- Dual-baseline Z-score anomaly detection (Individual vs Peer-Role)
- Markov Chain transition matrix & absorption time estimation
- Pearson correlation matrix
- Chi-Square test of independence
- SHAP feature attributions / Logistic Regression explainability
"""

import os
import csv
import math
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

STATES = ["NORMAL", "SUSPICIOUS", "HIGH RISK"]

FEATURE_NAMES = [
    "file_access",
    "login_count",
    "usb_activity",
    "email_count"
]

FEATURE_LABELS = {
    "file_access": "File Access",
    "login_count": "Login Count",
    "usb_activity": "USB Activity",
    "email_count": "Email Count"
}


def get_default_csv_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data", "employee_history.csv")
    if os.path.exists(data_path):
        return data_path
    fallback = r"C:\Users\manik\Documents\Insider_Threat_Math_Project\employee_history.csv"
    if os.path.exists(fallback):
        return fallback
    return "employee_history.csv"


def load_history(csv_path=None):
    if not csv_path:
        csv_path = get_default_csv_path()
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


def assign_historical_state(file_access, mean, std_dev):
    if std_dev <= 0:
        return "NORMAL"
    z = (file_access - mean) / std_dev
    if z < 1.0:
        return "NORMAL"
    elif z < 2.0:
        return "SUSPICIOUS"
    else:
        return "HIGH RISK"


def add_historical_risk_states(history):
    # Group by employee
    records_by_emp = {}
    for row in history:
        emp = row["employee_id"]
        records_by_emp.setdefault(emp, []).append(row)

    updated_history = []
    for emp, records in records_by_emp.items():
        # Sort by day
        records = sorted(records, key=lambda r: int(r.get("day", 1)))
        file_vals = [float(r["file_access"]) for r in records]
        mean = sum(file_vals) / len(file_vals) if file_vals else 0.0
        if len(file_vals) > 1:
            var = sum((v - mean) ** 2 for v in file_vals) / (len(file_vals) - 1)
            sd = math.sqrt(var)
        else:
            sd = 0.0

        for r in records:
            r_copy = dict(r)
            r_copy["risk_state"] = assign_historical_state(float(r["file_access"]), mean, sd)
            updated_history.append(r_copy)

    return updated_history


def calculate_transition_matrix(employee_history):
    """
    Computes 3x3 transition probability matrix between
    NORMAL (0), SUSPICIOUS (1), HIGH RISK (2).
    """
    matrix = [[0.0 for _ in range(3)] for _ in range(3)]
    counts = [[0 for _ in range(3)] for _ in range(3)]

    # Sort records by day
    sorted_history = sorted(employee_history, key=lambda r: int(r.get("day", 1)))

    for i in range(len(sorted_history) - 1):
        s_curr = sorted_history[i].get("risk_state", "NORMAL")
        s_next = sorted_history[i + 1].get("risk_state", "NORMAL")
        if s_curr in STATES and s_next in STATES:
            i_curr = STATES.index(s_curr)
            i_next = STATES.index(s_next)
            counts[i_curr][i_next] += 1

    for i in range(3):
        row_sum = sum(counts[i])
        if row_sum > 0:
            for j in range(3):
                matrix[i][j] = counts[i][j] / row_sum
        else:
            # Baseline stationary fallback
            if i == 0:
                matrix[i] = [0.70, 0.20, 0.10]
            elif i == 1:
                matrix[i] = [0.25, 0.50, 0.25]
            else:
                matrix[i] = [0.05, 0.25, 0.70]

    return matrix


def calculate_expected_time_to_high_risk(transition_matrix, current_state):
    """
    Calculates expected steps to reach absorbing state HIGH RISK (index 2)
    using the Fundamental Matrix: N = (I - Q)^(-1), t = N * 1
    """
    if current_state == "HIGH RISK":
        return 0.0

    try:
        # Transient states: NORMAL (0), SUSPICIOUS (1)
        Q = np.array([
            [transition_matrix[0][0], transition_matrix[0][1]],
            [transition_matrix[1][0], transition_matrix[1][1]]
        ], dtype=float)

        I = np.eye(2)
        diff = I - Q
        det = np.linalg.det(diff)
        if abs(det) < 1e-6:
            return None

        N = np.linalg.inv(diff)
        steps = N.dot(np.ones((2, 1))).flatten()

        if current_state == "NORMAL":
            return float(steps[0])
        elif current_state == "SUSPICIOUS":
            return float(steps[1])
        return None
    except Exception:
        return None


def calculate_correlations(history, current_record=None):
    """
    Calculates Pearson correlation matrix between the 4 telemetry features.
    """
    records = list(history)
    if current_record:
        records.append(current_record)

    data = {
        col: [float(r[col]) for r in records]
        for col in FEATURE_NAMES
    }

    names = [FEATURE_LABELS[k] for k in FEATURE_NAMES]
    matrix = []

    for name1, key1 in zip(names, FEATURE_NAMES):
        row_vals = []
        vals1 = data[key1]
        m1 = sum(vals1) / len(vals1) if vals1 else 0.0
        dev1 = [v - m1 for v in vals1]

        for name2, key2 in zip(names, FEATURE_NAMES):
            vals2 = data[key2]
            m2 = sum(vals2) / len(vals2) if vals2 else 0.0
            dev2 = [v - m2 for v in vals2]

            num = sum(a * b for a, b in zip(dev1, dev2))
            den = (sum(a ** 2 for a in dev1) * sum(b ** 2 for b in dev2)) ** 0.5
            corr = (num / den) if den != 0 else 0.0
            row_vals.append(round(corr, 3))

        matrix.append(row_vals)

    return {
        "features": names,
        "feature_keys": FEATURE_NAMES,
        "correlation_matrix": matrix
    }


def calculate_chi_square(history):
    """
    Chi-Square Test of Independence for elevated activities vs risk states.
    """
    try:
        history_with_states = add_historical_risk_states(history)
        # Build contingency table of state vs activity presence
        observed = []
        for state in STATES:
            state_records = [r for r in history_with_states if r.get("risk_state") == state]
            if not state_records:
                observed.append([1, 1, 1, 1])
                continue
            row = [
                sum(1 for r in state_records if float(r["file_access"]) > 10),
                sum(1 for r in state_records if float(r["login_count"]) > 6),
                sum(1 for r in state_records if float(r["usb_activity"]) > 0),
                sum(1 for r in state_records if float(r["email_count"]) > 10)
            ]
            # avoid all zeros
            row = [max(1, v) for v in row]
            observed.append(row)

        chi2_stat, p_val, dof, _ = chi2_contingency(observed)
        return {
            "chi2_stat": round(float(chi2_stat), 4),
            "p_value": round(float(p_val), 5),
            "dof": int(dof),
            "is_significant": bool(p_val < 0.05)
        }
    except Exception as e:
        return {
            "chi2_stat": 0.0,
            "p_value": 1.0,
            "dof": 6,
            "is_significant": False,
            "error": str(e)
        }


def calculate_shap_explanation(history, current_input):
    """
    Calculates feature importances/attributions for current input using Logistic Regression.
    """
    try:
        history_with_states = add_historical_risk_states(history)
        X = np.array([
            [float(r[col]) for col in FEATURE_NAMES]
            for r in history_with_states
        ])
        y = np.array([
            STATES.index(r["risk_state"])
            for r in history_with_states
        ])

        if len(set(y)) < 2:
            return {
                "features": [FEATURE_LABELS[k] for k in FEATURE_NAMES],
                "values": [0.25, 0.25, 0.25, 0.25],
                "predicted_class": "NORMAL"
            }

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_scaled, y)

        input_arr = np.array([[float(current_input[k]) for k in FEATURE_NAMES]])
        input_scaled = scaler.transform(input_arr)
        pred_class_idx = model.predict(input_scaled)[0]
        predicted_class = STATES[pred_class_idx]

        # Use absolute coefficients * scaled deviation as feature attributions
        if len(model.classes_) > 2:
            class_coef = model.coef_[pred_class_idx]
        else:
            class_coef = model.coef_[0] if pred_class_idx == 1 else -model.coef_[0]

        attributions = np.abs(class_coef * input_scaled[0])
        total = np.sum(attributions)
        if total > 0:
            normalized = attributions / total
        else:
            normalized = np.array([0.25, 0.25, 0.25, 0.25])

        return {
            "features": [FEATURE_LABELS[k] for k in FEATURE_NAMES],
            "feature_keys": FEATURE_NAMES,
            "values": [round(float(v), 3) for v in normalized],
            "raw_attributions": [round(float(v), 3) for v in attributions],
            "predicted_class": predicted_class
        }
    except Exception as e:
        return {
            "features": [FEATURE_LABELS[k] for k in FEATURE_NAMES],
            "feature_keys": FEATURE_NAMES,
            "values": [0.35, 0.20, 0.30, 0.15],
            "predicted_class": "NORMAL",
            "error": str(e)
        }


def analyze_employee(employee_id, inputs, history=None):
    """
    Performs full mathematical risk assessment for an employee with given inputs:
    inputs = {
        'file_access': float,
        'login_count': float,
        'usb_activity': float,
        'email_count': float
    }
    """
    if history is None:
        history = load_history()

    history_with_states = add_historical_risk_states(history)
    employee_history = [
        r for r in history_with_states
        if r["employee_id"] == employee_id
    ]

    if not employee_history:
        # Default fallback if employee has no records
        role = "Analyst"
        peer_records = [r for r in history_with_states if r.get("role") == role]
    else:
        role = employee_history[0].get("role", "Analyst")
        peer_records = [r for r in history_with_states if r.get("role") == role]

    # Markov transition matrix
    if len(employee_history) >= 2:
        transition_matrix = calculate_transition_matrix(employee_history)
    else:
        transition_matrix = calculate_transition_matrix(history_with_states)

    activity_scores = {}
    for col in FEATURE_NAMES:
        curr_val = float(inputs.get(col, 0.0))

        # Own baseline
        own_vals = [float(r[col]) for r in employee_history] if employee_history else [curr_val]
        own_mean = sum(own_vals) / len(own_vals)
        if len(own_vals) > 1:
            own_sd = math.sqrt(sum((v - own_mean) ** 2 for v in own_vals) / (len(own_vals) - 1))
        else:
            own_sd = 1.0
        own_sd = max(own_sd, 0.5)

        # Peer baseline
        peer_vals = [float(r[col]) for r in peer_records] if peer_records else own_vals
        peer_mean = sum(peer_vals) / len(peer_vals)
        if len(peer_vals) > 1:
            peer_sd = math.sqrt(sum((v - peer_mean) ** 2 for v in peer_vals) / (len(peer_vals) - 1))
        else:
            peer_sd = 1.0
        peer_sd = max(peer_sd, 0.5)

        own_z = (curr_val - own_mean) / own_sd
        peer_z = (curr_val - peer_mean) / peer_sd

        activity_scores[col] = {
            "label": FEATURE_LABELS[col],
            "current": curr_val,
            "own_mean": round(own_mean, 2),
            "own_sd": round(own_sd, 2),
            "peer_mean": round(peer_mean, 2),
            "peer_sd": round(peer_sd, 2),
            "self_z": round(own_z, 2),
            "peer_z": round(peer_z, 2),
            "max_z": round(max(own_z, peer_z), 2)
        }

    # Find highest Z-score across features
    highest_col = max(activity_scores, key=lambda k: activity_scores[k]["max_z"])
    highest_info = activity_scores[highest_col]
    highest_z = highest_info["max_z"]

    if highest_z < 1.0:
        risk_level = "LOW"
        risk_state = "NORMAL"
        severity = "Low"
        color = "#22C55E"
    elif highest_z < 2.0:
        risk_level = "MEDIUM"
        risk_state = "SUSPICIOUS"
        severity = "Medium"
        color = "#F59E0B"
    elif highest_z < 3.0:
        risk_level = "HIGH"
        risk_state = "HIGH RISK"
        severity = "High"
        color = "#F97316"
    else:
        risk_level = "CRITICAL"
        risk_state = "HIGH RISK"
        severity = "Critical"
        color = "#EF4444"

    # Numeric risk score normalized 0 to 100
    risk_score = min(100, max(5, int(15 + highest_z * 25)))

    expected_time = calculate_expected_time_to_high_risk(transition_matrix, risk_state)

    # Baseline finding explanation
    self_z = highest_info["self_z"]
    peer_z = highest_info["peer_z"]
    if self_z >= 2 and peer_z >= 2:
        finding = f"{highest_info['label']}: Both baselines flag anomalous activity"
    elif self_z >= 2:
        finding = f"{highest_info['label']}: Unusual compared with employee's own history"
    elif peer_z >= 2:
        finding = f"{highest_info['label']}: Unusual compared with peer group"
    elif max(self_z, peer_z) >= 1:
        finding = f"{highest_info['label']}: Activity is elevated above baseline"
    else:
        finding = f"{highest_info['label']}: Nominal activity within expected baseline"

    # SHAP explanations
    shap_data = calculate_shap_explanation(history, inputs)

    # Next state probabilities from transition matrix
    state_idx = STATES.index(risk_state)
    next_state_probs = {
        STATES[j]: round(transition_matrix[state_idx][j], 3)
        for j in range(3)
    }

    return {
        "employee_id": employee_id,
        "role": role,
        "risk_level": risk_level,
        "risk_state": risk_state,
        "risk_score": risk_score,
        "severity": severity,
        "color": color,
        "highest_activity": highest_info["label"],
        "highest_z": highest_z,
        "finding": finding,
        "activity_scores": activity_scores,
        "transition_matrix": transition_matrix,
        "expected_time_to_high_risk": round(expected_time, 2) if expected_time is not None else None,
        "next_state_probs": next_state_probs,
        "shap": shap_data
    }
