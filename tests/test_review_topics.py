from __future__ import annotations

import pytest

from chess_scan.review_detectors import DETECTABLE_HANDLERS
from chess_scan.review_topics import DEFAULT_TOPIC, FORK_TOPICS, REVIEW_TOPICS, topic_for


def test_every_detectable_subject_has_human_authored_copy() -> None:
    assert set(REVIEW_TOPICS) == DETECTABLE_HANDLERS
    topics = [*REVIEW_TOPICS.values(), *FORK_TOPICS.values()]

    assert len({topic.id for topic in topics}) == len(topics)
    assert all(topic.name and topic.hint and topic.idea for topic in topics)
    assert all("stockfish" not in topic.hint.lower() for topic in topics)


def test_piece_specific_forks_use_plain_chess_terms() -> None:
    assert topic_for("double_attack", "two_targets_knight").name == "Knight fork"
    assert topic_for("double_attack", "two_targets_pawn").name == "Pawn fork"
    assert topic_for("double_attack", "two_targets_queen").name == "Double attack"


def test_topic_copy_matches_what_position_evaluators_prove() -> None:
    check = topic_for("check")
    pawn_race = topic_for("pawn_race")

    assert check.name == "Giving check"
    assert "opponent must answer" in check.idea
    assert "wins the race" not in pawn_race.idea


def test_no_finding_uses_the_default_and_unknown_handlers_fail() -> None:
    assert topic_for(None) is DEFAULT_TOPIC
    with pytest.raises(ValueError, match="Unknown review topic handler"):
        topic_for("not-supported")
