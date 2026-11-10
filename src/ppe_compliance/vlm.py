"""Short natural-language infraction descriptions with a vision-language model.

The model (Idefics2-8b by default) is loaded lazily on first use, so the app
and the headless analysis can run without it (``--no-vlm``). Any model
supported by ``AutoModelForImageTextToText`` can be selected with
``--vlm-model``, e.g. the CPU-friendly ``HuggingFaceTB/SmolVLM-500M-Instruct``.
When the processor has a chat template it is used to build the prompt;
otherwise an Idefics2-style ``User: <image> ...`` prompt is used.
"""

import logging
from typing import Any, Optional, Sequence

from .config import VLMConfig

logger = logging.getLogger(__name__)

#: Instruction sent with the person crop; ``{ppes}`` is the list of flagged labels.
PROMPT_TEXT = (
    "This is a cropped image of an industrial worker. The detection system flagged the "
    "following missing PPE: {ppes}. Double-check the image and answer in one short "
    "sentence whether you confirm the missing PPE or not."
)

#: Idefics2-style prompt used when the processor has no chat template.
PROMPT_TEMPLATE = "User: <image> " + PROMPT_TEXT + "\nAssistant:"


def clean_answer(text: str) -> str:
    """Strip whitespace and any ``Assistant:`` prefix from generated text.

    Args:
        text: Decoded model output.

    Returns:
        The answer without role prefixes.
    """
    text = text.strip()
    if "Assistant:" in text:
        text = text.split("Assistant:")[-1].strip()
    return text


def fallback_description(missing_ppes: Sequence[str]) -> str:
    """Return the template sentence used when no VLM text is available.

    Args:
        missing_ppes: Infraction labels.

    Returns:
        A one-sentence description.
    """
    return f"Worker operating without the following mandatory safety items: {', '.join(missing_ppes)}."


class VLMReporter:
    """Describes a person crop and its flagged infractions in one sentence."""

    def __init__(self, config: Optional[VLMConfig] = None) -> None:
        """Store the configuration; the model is not loaded yet.

        Args:
            config: VLM settings; defaults are used when ``None``.
        """
        self.config = config or VLMConfig()
        self._processor: Any = None
        self._model: Any = None
        self._device = "cpu"

    @property
    def loaded(self) -> bool:
        """Whether the model is already in memory."""
        return self._model is not None

    def load(self) -> None:
        """Load the processor and the model (no-op if already loaded).

        Raises:
            RuntimeError: If the dependencies or the model cannot be loaded.
        """
        if self.loaded:
            return
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "The VLM report needs 'torch' and 'transformers'. "
                "Install the requirements or run with --no-vlm."
            ) from exc

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device == "cpu":
            logger.warning("No CUDA GPU found: the VLM will run on CPU and be very slow.")
        model_id = self.config.model_id
        try:
            logger.info("Loading VLM %s on %s", model_id, self._device)
            self._processor = AutoProcessor.from_pretrained(model_id)
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            ).to(self._device)
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            self._processor = self._model = None
            raise RuntimeError(
                f"Failed to load VLM '{model_id}': {exc}. A large GPU is recommended; "
                "check available memory and network access, or run with --no-vlm."
            ) from exc

    def _build_prompt(self, ppes: str) -> str:
        """Build the text prompt, with the processor's chat template when it has one.

        Args:
            ppes: Comma-separated infraction labels.

        Returns:
            The prompt string containing one image placeholder.
        """
        if getattr(self._processor, "chat_template", None):
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT_TEXT.format(ppes=ppes)},
            ]}]
            return self._processor.apply_chat_template(messages, add_generation_prompt=True)
        return PROMPT_TEMPLATE.format(ppes=ppes)

    def describe(self, crop_bgr: Any, missing_ppes: Sequence[str]) -> str:
        """Generate a one-sentence description of a person's infractions.

        Args:
            crop_bgr: Person crop as a BGR image array.
            missing_ppes: Infraction labels flagged by the detector.

        Returns:
            The generated sentence, or :func:`fallback_description` if the
            model produced (almost) no text.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        self.load()
        import torch

        from .drawing import bgr_to_pil

        prompt = self._build_prompt(", ".join(missing_ppes))
        inputs = self._processor(
            text=[prompt], images=[bgr_to_pil(crop_bgr)], padding=True, return_tensors="pt"
        )
        dtype = self._model.dtype  # fp16 on CUDA: pixel values must match the weights
        inputs = {
            k: v.to(self._device, dtype=dtype) if v.is_floating_point() else v.to(self._device)
            for k, v in inputs.items()
        }
        tokenizer = self._processor.tokenizer
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=self.config.repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        text = clean_answer(self._processor.decode(generated, skip_special_tokens=True))
        if len(text) < 3:
            return fallback_description(missing_ppes)
        return text
