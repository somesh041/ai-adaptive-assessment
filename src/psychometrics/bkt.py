from __future__ import annotations


def bkt_update(p_known: float, correct: bool, slip: float = 0.10, guess: float = 0.20, learn: float = 0.15) -> float:
    """
    Classic Bayesian Knowledge Tracing update (single skill):
      - p_known: P(Ln) before observing response
      - slip: P(incorrect | known)
      - guess: P(correct | not known)
      - learn: P(transition to known after opportunity)

    Returns P(Ln+1) after observation + learning transition.
    """
    p_known = min(1.0, max(0.0, p_known))

    if correct:
        num = p_known * (1.0 - slip)
        den = num + (1.0 - p_known) * guess
    else:
        num = p_known * slip
        den = num + (1.0 - p_known) * (1.0 - guess)

    posterior = num / max(1e-12, den)

    # learning transition
    posterior = posterior + (1.0 - posterior) * learn
    return float(min(1.0, max(0.0, posterior)))
