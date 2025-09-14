"""
Common configuration constants for the gym recognition project.
"""

# Windowing: 3 seconds * 20 Hz = 60 samples per window
WINDOW_SECONDS = 3
FS = 20
WINDOW_SIZE = WINDOW_SECONDS * FS

# 50% overlap (slide by 1.5 seconds)
WINDOW_STRIDE = WINDOW_SIZE // 2

# Expected sensor columns (will be auto-detected from available columns)
# RecGym format: IMU + body capacitance
SENSOR_COLS = ["A_x", "A_y", "A_z", "G_x", "G_y", "G_z", "C_1"]

# Candidate column names for labels and group/subject IDs (dataset may vary)
LABEL_CANDIDATES = ["Workout", "workout", "activity", "Activity", "label", "Label", "class", "Class", "exercise", "Exercise"]
GROUP_CANDIDATES = ["Object", "object", "volunteer", "Volunteer", "subject", "Subject", "user_name", "user", "User", "participant", "Participant"]

# For reproducible splits
RANDOM_STATE = 42