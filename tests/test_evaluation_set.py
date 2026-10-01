from src.evaluation_set import ON_TOPIC, OFF_TOPIC, all_cases


def test_case_counts():
    assert len(ON_TOPIC) == 6
    assert len(OFF_TOPIC) == 5


def test_all_cases_tags_kind():
    cases = all_cases()
    assert len(cases) == 11
    assert sum(c["kind"] == "on_topic" for c in cases) == 6
    assert sum(c["kind"] == "off_topic" for c in cases) == 5


def test_off_topic_includes_both_injection_attempts():
    queries = [c["query"] for c in OFF_TOPIC]
    assert any("ignore your" in q.lower() for q in queries)
    assert any("pirate" in q.lower() for q in queries)