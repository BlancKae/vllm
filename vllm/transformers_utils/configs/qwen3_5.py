"""Minimal local Qwen3.5 text/vision configuration bridge.

Transformers 4.56 used by the vLLM environment predates Qwen3.5.  The
language configuration intentionally mirrors Qwen3Next's fields so the vLLM
model config parser can inspect nested ``text_config`` without changing the
installed Transformers package.
"""
from .qwen3_next import Qwen3NextConfig


class Qwen3_5TextConfig(Qwen3NextConfig):
    model_type = "qwen3_5_text"

    def __init__(self, **kwargs):
        rope = kwargs.pop("rope_parameters", None) or {}
        # Qwen3.5 stores RoPE fields in rope_parameters, unlike Qwen3Next.
        if "rope_theta" in rope:
            kwargs.setdefault("rope_theta", rope["rope_theta"])
        if "partial_rotary_factor" in rope:
            kwargs.setdefault("partial_rotary_factor", rope["partial_rotary_factor"])
        if rope.get("rope_type", "default") != "default":
            kwargs.setdefault("rope_scaling", rope)
        super().__init__(**kwargs)
        if not hasattr(self, "layer_types"):
            self.layer_types = kwargs.get("layer_types", [])


class Qwen3_5Config(Qwen3NextConfig):
    model_type = "qwen3_5"

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        self.text_config = (text_config if isinstance(text_config, Qwen3_5TextConfig)
                            else Qwen3_5TextConfig(**(text_config or {})))
        self.vision_config = vision_config or {}
        super().__init__(**self.text_config.to_dict())
        self.model_type = "qwen3_5"


__all__ = ["Qwen3_5Config", "Qwen3_5TextConfig"]
