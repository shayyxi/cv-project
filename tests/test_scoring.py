from app.analytics.scoring_service import weighted_score

SEVERITY = {"helmet": 3.0, "vest": 2.0, "boots": 1.0}


def test_no_violations_scores_100() -> None:
    assert weighted_score(5, 0, 0, 0, SEVERITY) == 100.0


def test_everything_missing_scores_0() -> None:
    assert weighted_score(2, 2, 2, 2, SEVERITY) == 0.0


def test_helmet_weighs_more_than_boots() -> None:
    helmet_only = weighted_score(1, 1, 0, 0, SEVERITY)
    boots_only = weighted_score(1, 0, 0, 1, SEVERITY)

    assert helmet_only < boots_only


def test_known_example() -> None:
    # 1 worker missing vest (2) and boots (1) out of weight 6 -> 50.0
    assert weighted_score(1, 0, 1, 1, SEVERITY) == 50.0


def test_no_workers_scores_100() -> None:
    assert weighted_score(0, 0, 0, 0, SEVERITY) == 100.0
