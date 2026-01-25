from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Optional


@contextmanager
def hidiffusion_patch(
    model: Any,
    name_or_path: Optional[str],
    apply_raunet: bool = True,
    apply_window_attn: bool = True,
):
    """Context manager that applies HiDiffusion and restores the model on exit."""
    try:
        from hidiffusion import apply_hidiffusion, remove_hidiffusion
    except ImportError as exc:
        raise ImportError(
            "HiDiffusion is not installed. Install it with `pip install hidiffusion` to enable this option."
        ) from exc

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

    set_model_name_or_path = False
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

    original_num_upsamplers = getattr(target, "num_upsamplers", None)

    apply_hidiffusion(model, apply_raunet=apply_raunet, apply_window_attn=apply_window_attn)
    try:
        yield
    finally:
        remove_hidiffusion(model)
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
            else:
                if config is not None:
                    try:
                        delattr(config, "_name_or_path")
                    except AttributeError:
                        pass
