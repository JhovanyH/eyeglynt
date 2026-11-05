"""Dropdown options — must match the ENUMs in database/schema.sql."""

ROLES = [
    "Family Member",
    "Personal Caregiver",
    "Therapist",
    "Researcher",
    "Other",
]

CARE_SETTINGS = ["Home", "Hospital", "Rehabilitation / Therapy Center", "Long-term Care Facility", "Other"]

# (milliseconds, label). The device uses the dwell time chosen here for the
# whole session. The default is the device's own DWELL_TIME (main.py): in
# testing, 1.5 s and 2 s were too quick for the gaze pointer's accuracy.
DWELL_OPTIONS = [
    (1500, "1.5 seconds"),
    (2000, "2.0 seconds"),
    (2500, "2.5 seconds"),
    (3000, "3.0 seconds (default)"),
    (3500, "3.5 seconds"),
    (4000, "4.0 seconds"),
]
DEFAULT_DWELL_MS = 3000

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
    "calibration_required": "The device needs to be calibrated.",
}
