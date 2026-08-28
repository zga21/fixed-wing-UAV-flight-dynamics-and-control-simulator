"""Fast Phase 6 candidate-screen tests."""

from scripts.ndi_screen import screen_candidate


def test_screen_candidate_is_deterministic():
    first = screen_candidate(0.2, 0.5, 0.05, seeds=(3,))
    second = screen_candidate(0.2, 0.5, 0.05, seeds=(3,))

    assert first == second
    assert first["ndi_score"] > 0.0
    assert first["pid_score"] > 0.0
