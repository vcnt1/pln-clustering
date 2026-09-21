import pytest

from transform.features import extract_features


def _msg(n: int, role: str = "customer") -> dict:
    return {
        "message_id": f"syn-msg-{n:06d}",
        "role": role,
        "text": f"mensagem numero {n}",
        "sent_at": f"2026-01-01T10:{n:02d}:00Z",
    }


def test_single_message_history_has_empty_context() -> None:
    result = extract_features([_msg(1)])
    assert result["text_clean"] == "mensagem numero 1"
    assert result["context_clean"] == []


def test_thirty_item_history_does_not_raise() -> None:
    history = [_msg(i) for i in range(30)]
    result = extract_features(history)
    assert len(result["context_clean"]) == 29
    assert result["text_clean"] == "mensagem numero 29"


def test_empty_history_raises_value_error() -> None:
    with pytest.raises(ValueError):
        extract_features([])


def test_history_over_thirty_items_raises_value_error() -> None:
    with pytest.raises(ValueError):
        extract_features([_msg(i) for i in range(31)])


def test_agent_role_in_history_raises_value_error() -> None:
    history = [_msg(1), _msg(2, role="agent")]
    with pytest.raises(ValueError):
        extract_features(history)


def test_context_clean_preserves_chronological_order() -> None:
    history = [_msg(1), _msg(2), _msg(3)]
    result = extract_features(history)
    assert result["context_clean"] == ["mensagem numero 1", "mensagem numero 2"]
    assert result["text_clean"] == "mensagem numero 3"


def test_t1_and_t2_are_applied_to_every_item_not_only_the_trigger() -> None:
    history = [
        {**_msg(1), "text": "Meu email e a@example.com"},
        {**_msg(2), "text": "   texto   com    espacos  "},
    ]
    result = extract_features(history)
    assert result["context_clean"] == ["Meu email e <EMAIL>"]
    assert result["text_clean"] == "texto com espacos"


def test_extract_features_is_deterministic_across_calls() -> None:
    history = [_msg(1), _msg(2)]
    first = extract_features(history)
    second = extract_features(history)
    assert first == second
