"""FinQA terminal-risk evaluation study."""

CANDIDATES = 8
CONFIDENCES = 9
STATES = 1 + CANDIDATES * CONFIDENCES  # root + (candidate, confidence)
OUTCOMES = CANDIDATES * CONFIDENCES
CANDIDATE_LABELS = tuple("ABCDEFGH")
CONFIDENCE_LABELS = tuple(str(i) for i in range(1, 10))
WORKFLOWS = ("ordinary", "unitcheck", "skeptical")
HORIZON = 3
