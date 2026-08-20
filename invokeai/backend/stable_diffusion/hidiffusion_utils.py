from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any, Optional

import torch


@contextmanager
def hidiffusion_patch(
    model: Any,
    name_or_path: Optional[str],
    apply_raunet: bool = True,
    apply_window_attn: bool = True,
    t1_ratio: Optional[float] = None,
    t2_ratio: Optional[float] = None,
    generator: torch.Generator | None = None,
    has_controlnet: bool = False,
    is_controlnet_text_to_image: bool = False,
):
    """Context manager that applies HiDiffusion and restores the model on exit."""
    from invokeai.backend.hidiffusion.hidiffusion import apply_hidiffusion, remove_hidiffusion

    target = model.unet if hasattr(model, "unet") else model

    had_model_name_or_path = hasattr(model, "_name_or_path")
    had_config = hasattr(model, "config")
    config = model.config if had_config else None
    had_config_name_or_path = bool(config) and hasattr(config, "_name_or_path")

    original_model_name_or_path = model._name_or_path if had_model_name_or_path else None
    original_config_name_or_path = config._name_or_path if had_config_name_or_path else None

    effective_name_or_path = (
        name_or_path
        or getattr(model, "name_or_path", None)
        or original_model_name_or_path
        or original_config_name_or_path
        or ""
    )

    def _set_name_or_path_on_config(cfg, value: str) -> bool:
        if cfg is None:
            return False
        if hasattr(cfg, "_internal_dict"):
            try:
                cfg._internal_dict["_name_or_path"] = value
                return True
            except Exception:
                pass
        try:
            object.__setattr__(cfg, "_name_or_path", value)
            return True
        except Exception:
            pass
        try:
            cfg.__dict__["_name_or_path"] = value
            return True
        except Exception:
            return False

    original_num_upsamplers = getattr(target, "num_upsamplers", None)

    set_model_name_or_path = False
    set_config_name_or_path = False
    try:
        try:
            object.__setattr__(model, "_name_or_path", effective_name_or_path)
            set_model_name_or_path = True
        except Exception:
            set_model_name_or_path = False

        set_config_name_or_path = _set_name_or_path_on_config(config, effective_name_or_path)

        # Ensure the property resolves to a non-None value before calling HiDiffusion.
        try:
            if getattr(model, "name_or_path", None) is None:
                if not set_model_name_or_path:
                    try:
                        object.__setattr__(model, "_name_or_path", effective_name_or_path)
                        set_model_name_or_path = True
                    except Exception:
                        pass
                if not set_config_name_or_path:
                    set_config_name_or_path = _set_name_or_path_on_config(config, effective_name_or_path)
        except Exception:
            pass

        apply_hidiffusion(
            model,
            apply_raunet=apply_raunet,
            apply_window_attn=apply_window_attn,
            t1_ratio=t1_ratio,
            t2_ratio=t2_ratio,
            has_controlnet=has_controlnet,
            is_controlnet_text_to_image=is_controlnet_text_to_image,
            generator=generator,
        )
        yield
    finally:
        had_active_exception = sys.exc_info()[0] is not None
        teardown_error: Exception | None = None
        try:
            remove_hidiffusion(model)
        except Exception as error:
            if not had_active_exception:
                teardown_error = error
        if original_num_upsamplers is not None:
            target.num_upsamplers = original_num_upsamplers
        if set_model_name_or_path:
            if had_model_name_or_path:
                try:
                    object.__setattr__(model, "_name_or_path", original_model_name_or_path)
                except Exception:
                    pass
            else:
                try:
                    delattr(model, "_name_or_path")
                except AttributeError:
                    pass
        if set_config_name_or_path and had_config:
            if had_config_name_or_path:
                _set_name_or_path_on_config(config, original_config_name_or_path)
            elif config is not None:
                internal_dict = getattr(config, "_internal_dict", None)
                if internal_dict is not None:
                    try:
                        internal_dict.pop("_name_or_path", None)
                    except Exception:
                        pass
                try:
                    delattr(config, "_name_or_path")
                except AttributeError:
                    pass
        if teardown_error is not None:
            raise teardown_error
