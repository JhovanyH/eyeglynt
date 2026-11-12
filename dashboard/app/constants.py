"""Dropdown options — must match the ENUMs in database/schema.sql."""

ROLES = [
    "Family Member",
    "Personal Caregiver",
    "Therapist",
    "Researcher",
    "Other",
]

CARE_SETTINGS = ["Home", "Hospital", "Rehabilitation / Therapy Center", "Long-term Care Facility", "Other"]

# Dwell time picked for a session on the Start page (milliseconds, label).
# The device uses it for the whole session, then goes back to its own
# DWELL_TIME (main.py, 4 s).
DWELL_OPTIONS = [
    (2000, "2 seconds"),
    (3000, "3 seconds"),
    (4000, "4 seconds (default)"),
    (5000, "5 seconds"),
    (6000, "6 seconds"),
]
DEFAULT_DWELL_MS = 4000

# Thesis Table 7 — Performance Benchmarks
BENCHMARKS = {
    "gaze_accuracy_pct":         {"label": "Gaze Accuracy",       "op": ">=", "value": 85,  "unit": "%",  "target": "Min 85%"},
    "false_positive_pct":        {"label": "False Positive Rate", "op": "<",  "value": 10,  "unit": "%",  "target": "Max 10%"},
    "avg_detection_latency_ms":  {"label": "Detection Latency",   "op": "<",  "value": 500, "unit": "ms", "target": "Max 500 ms"},
    "recalibration_success_pct": {"label": "Recal. Success",      "op": ">=", "value": 80,  "unit": "%",  "target": "Min 80%"},
}

ALERT_LABELS = {
    "lighting_shift": "Lighting shift detected",
    "gaze_lock_loss": "Gaze lock lost",
    "gaze_lock_restored": "Gaze lock restored",
    "recalibration_started": "Recalibration started",
    "recalibration_success": "Recalibration successful",
    "recalibration_failed": "Recalibration failed",
    "recalibration_cancelled": "Calibration stopped",
    "session_auto_ended": "EyeGlynt app closed",
    "calibration_required": "Calibration required",
    "system": "System",
}

# Plain-language alert text for caregivers (researchers see the technical message).
CAREGIVER_ALERT_TEXT = {
    "lighting_shift": "Room lighting changed. The device is adjusting.",
    "gaze_lock_loss": "The device can't see the patient's eyes right now.",
    "gaze_lock_restored": "The patient's eyes are detected again.",
    "recalibration_started": "The device is recalibrating.",
    "recalibration_success": "Recalibration finished. Tracking is back.",
    "recalibration_failed": "Recalibration didn't work. Check the patient's position and the room lighting.",
    "recalibration_cancelled": "Calibration was stopped. The device went back to the keyboard.",
    "session_auto_ended": "The EyeGlynt app was closed on the device, so the session ended.",
    "calibration_required": "The device needs to be calibrated.",
}
