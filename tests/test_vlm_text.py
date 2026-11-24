"""Tests for the pure text helpers of :mod:`ppe_compliance.reporting.vlm` (standard library only)."""

from ppe_compliance.reporting.vlm import PROMPT_TEMPLATE, PROMPT_TEXT, clean_answer, fallback_description


def test_clean_answer_strips_assistant_prefix() -> None:
    assert clean_answer("  Assistant: The worker has no hardhat. ") == "The worker has no hardhat."


def test_clean_answer_keeps_plain_text() -> None:
    assert clean_answer(" Yes, no vest is visible.\n") == "Yes, no vest is visible."


def test_clean_answer_uses_last_assistant_turn() -> None:
    assert clean_answer("User: hi\nAssistant: one\nAssistant: two") == "two"


def test_prompt_template_wraps_prompt_text() -> None:
    prompt = PROMPT_TEMPLATE.format(labels="NO-Hardhat")
    assert prompt.startswith("User: <image> ")
    assert prompt.endswith("\nAssistant:")
    assert PROMPT_TEXT.format(labels="NO-Hardhat") in prompt


def test_fallback_description_lists_items() -> None:
    assert "NO-Hardhat, NO-Mask" in fallback_description(["NO-Hardhat", "NO-Mask"])


def test_prompt_fits_event_labels_and_is_not_ppe_only() -> None:
    prompt = PROMPT_TEXT.format(labels="Fall-Detected, NO-Hardhat")
    assert "Fall-Detected, NO-Hardhat" in prompt
    assert "event" in prompt and "fall" in prompt
    assert "missing PPE" not in prompt  # would mislabel events as missing equipment


def test_describe_is_serialized_and_releases_the_lock_on_error() -> None:
    import threading

    from ppe_compliance.reporting.vlm import VLMReporter

    class Probe(VLMReporter):
        active = 0
        peak = 0
        fail = False

        def _describe_locked(self, crop_bgr, infractions):
            type(self).active += 1
            type(self).peak = max(type(self).peak, type(self).active)
            try:
                if type(self).fail:
                    raise RuntimeError("boom")
                threading.Event().wait(0.02)
                return "ok"
            finally:
                type(self).active -= 1

    reporter = Probe()
    threads = [threading.Thread(target=reporter.describe, args=(None, ["NO-Hardhat"])) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert Probe.peak == 1  # never two at once

    Probe.fail = True
    try:
        reporter.describe(None, [])
    except RuntimeError:
        pass
    assert reporter._lock.acquire(blocking=False)  # released after the error
    reporter._lock.release()
