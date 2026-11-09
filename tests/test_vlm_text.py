"""Tests for the pure text helpers of :mod:`ppe_compliance.vlm` (standard library only)."""

from ppe_compliance.vlm import PROMPT_TEMPLATE, PROMPT_TEXT, clean_answer, fallback_description


def test_clean_answer_strips_assistant_prefix() -> None:
    assert clean_answer("  Assistant: The worker has no hardhat. ") == "The worker has no hardhat."


def test_clean_answer_keeps_plain_text() -> None:
    assert clean_answer(" Yes, no vest is visible.\n") == "Yes, no vest is visible."


def test_clean_answer_uses_last_assistant_turn() -> None:
    assert clean_answer("User: hi\nAssistant: one\nAssistant: two") == "two"


def test_prompt_template_wraps_prompt_text() -> None:
    prompt = PROMPT_TEMPLATE.format(ppes="NO-Hardhat")
    assert prompt.startswith("User: <image> ")
    assert prompt.endswith("\nAssistant:")
    assert PROMPT_TEXT.format(ppes="NO-Hardhat") in prompt


def test_fallback_description_lists_items() -> None:
    assert "NO-Hardhat, NO-Mask" in fallback_description(["NO-Hardhat", "NO-Mask"])
