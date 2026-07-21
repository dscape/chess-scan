from __future__ import annotations

from collections import Counter

import pytest

from chess_scan.review_detectors import (
    AUTOMATIC_HANDLERS,
    HISTORY_HANDLERS,
    POLICY_HANDLERS,
    UNSUPPORTED_HANDLERS,
    registry_handlers_are_complete,
)
from chess_scan.review_topics import (
    REVIEW_TOPICS,
    TOPIC_REGISTRY_VERSION,
    TOPICS_BY_ID,
    Course,
    TopicCapability,
    topic_by_id,
    topics_for_level,
)


def test_topic_registry_is_exhaustive_unique_and_implemented() -> None:
    assert TOPIC_REGISTRY_VERSION == "chess-scan-curriculum-1"
    assert len(REVIEW_TOPICS) == 144
    assert len(TOPICS_BY_ID) == len(REVIEW_TOPICS)
    assert registry_handlers_are_complete()
    assert Counter((topic.level, topic.course) for topic in REVIEW_TOPICS) == {
        (1, Course.BASIC): 15,
        (1, Course.PLUS): 8,
        (2, Course.BASIC): 13,
        (2, Course.PLUS): 9,
        (3, Course.BASIC): 17,
        (3, Course.PLUS): 14,
        (4, Course.BASIC): 17,
        (4, Course.PLUS): 11,
        (5, Course.BASIC): 16,
        (5, Course.PLUS): 10,
        (6, Course.BASIC): 14,
    }
    handler_groups = [
        AUTOMATIC_HANDLERS,
        HISTORY_HANDLERS,
        POLICY_HANDLERS,
        UNSUPPORTED_HANDLERS,
    ]
    assert set.union(*handler_groups) == {topic.handler for topic in REVIEW_TOPICS}
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(handler_groups)
        for right in handler_groups[index + 1 :]
    )
    assert all(topic.id == topic.id.lower() for topic in REVIEW_TOPICS)
    assert all("chess steps" not in topic.name.lower() for topic in REVIEW_TOPICS)


def test_topic_registry_separates_detectors_policies_and_history() -> None:
    capabilities = Counter(topic.capability for topic in REVIEW_TOPICS)

    assert capabilities[TopicCapability.DETECTOR] > 40
    assert capabilities[TopicCapability.EVALUATOR] > 25
    assert capabilities[TopicCapability.POLICY] > 5
    assert capabilities[TopicCapability.HISTORY_REQUIRED] == 7
    assert capabilities[TopicCapability.UNSUPPORTED] == 16
    assert topic_by_id("level-2.basic.pin").handler == "pin"
    assert topic_by_id("level-2.plus.opening").capability is TopicCapability.HISTORY_REQUIRED
    assert topic_by_id("level-4.basic.thinking-ahead").capability is TopicCapability.POLICY
    assert topic_by_id("level-5.plus.zugzwang").capability is TopicCapability.UNSUPPORTED


def test_topics_for_level_never_leaks_later_or_plus_material() -> None:
    level_two = topics_for_level(2, include_plus=False)

    assert level_two
    assert all(topic.level <= 2 for topic in level_two)
    assert all(topic.course is Course.BASIC for topic in level_two)
    assert "level-2.basic.pin" in {topic.id for topic in level_two}
    assert "level-3.basic.x-ray" not in {topic.id for topic in level_two}

    with pytest.raises(ValueError, match="between 1 and 6"):
        topics_for_level(0)
    with pytest.raises(KeyError, match="Unknown review topic"):
        topic_by_id("missing")
