"""Every loader that could be offered FP8 Storage either implements it or says why not.

The control was derived from the model type alone, and honoured by whichever loader happened to call
the cast. Nothing connected the two, so 19 of the 52 offered loader keys did nothing with it -- and
identification switched it on by itself for a float8 denoiser, which wrote `fp8_storage: true` onto
records whose loader ignores it.

The walk below is the part that cannot rot: a loader added tomorrow with neither the cast nor a
declaration fails here, named. It reads the loaders' source to decide whether they implement the cast,
which is only sound while every call is `self._apply_fp8_layerwise_casting(...)` inside a loader
class -- so that assumption is a cell of its own rather than a comment.
"""

import ast
import inspect
import textwrap
from pathlib import Path

import pytest

import invokeai.backend.model_manager.load  # noqa: F401  (registers every loader)
from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig  # noqa: F401  (builds every config class)
from invokeai.backend.model_manager.load import fp8_capability
from invokeai.backend.model_manager.load.fp8_capability import (
    FP8_STORAGE_MODEL_TYPES,
    NotApplicable,
    Unimplemented,
    declare_fp8_storage,
    declared_fp8_storage,
    fp8_storage_support,
    fp8_storage_verdict,
)
from invokeai.backend.model_manager.load.load_base import ModelLoaderBase
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import (
    QUANTIZED_MODEL_FORMATS,
    BaseModelType,
    ModelFormat,
    ModelType,
)

_CAST = "_apply_fp8_layerwise_casting"
_KEEP = "_keep_fp8_weights"
_HELPERS = (_CAST, _KEEP)
# `_should_use_fp8` is deliberately *not* one of these, although `qwen_image.py` does consult it to
# decide whether to keep its packed weights. Consulting the gate is not honouring the setting: `wan.py`
# asks it only to decide whether to explain a decline, so counting it made this walk pass a Wan
# checkpoint whose cast had been deleted -- measured, by mutation. A loader that honoured the setting
# only that way would instead be asked for a declaration it does not need, which fails loudly and in
# the safe direction; a false positive is how `WanDiffusersModel` stayed hidden for as long as it did.
_LOADERS_DIR = Path(inspect.getfile(invokeai.backend.model_manager.load)).parent / "model_loaders"

# The two classes that *define* the fp8 helpers. Their source is excluded from the scan below, or
# every loader would look like it implements the cast by inheriting the method that performs it.
_GENERIC = (ModelLoader, ModelLoaderBase, object)


def _undeclare_for_test(key: tuple[BaseModelType, ModelType, ModelFormat]) -> None:
    """Remove a declaration this file added. The module has no remover, because production never needs one."""
    fp8_capability._declared.pop(key)


def _self_calls(source: str) -> set[str]:
    """The names a piece of source calls as `self.<name>(...)`.

    Parsed rather than grepped: `qwen_image.py` and `z_image.py` both discuss
    `_apply_fp8_layerwise_casting` in comments, and a substring scan reads those as implementations --
    the exact kind of false pass this file exists to prevent.
    """
    return {
        node.func.attr
        for node in ast.walk(ast.parse(textwrap.dedent(source)))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    }


def _super_calls(source: str) -> set[str]:
    """The names a piece of source calls as `super().<name>(...)`."""
    return {
        node.func.attr
        for node in ast.walk(ast.parse(textwrap.dedent(source)))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "super"
    }


def _method_source(klass: type, name: str) -> str:
    """The source of the method `klass` itself defines, unwrapped from any descriptor."""
    attribute = vars(klass)[name]
    return inspect.getsource(getattr(attribute, "__func__", attribute))


def _implements_the_cast(loader_cls: type) -> bool:
    """Whether the `_load_model` that would actually run reaches FP8 Storage.

    Either by calling the cast, or by asking `_keep_fp8_weights` -- a checkpoint loader that keeps
    scaled fp8 weights packed when the setting is on has honoured it just as much, without ever casting
    anything.

    Neither of the two obvious rules works, and both failed against real loaders in this tree:

    - scanning every class in the MRO credits an inherited call that an override replaced.
      `WanDiffusersModel` overrides `_load_model` outright and never delegates, so
      `GenericDiffusersLoader`'s cast never runs -- yet a flat scan called it implemented, and that
      false pass hid a live dead toggle;
    - scanning only the class that defines `_load_model` misses work handed to a helper elsewhere in
      the MRO. LTX-2 puts `_load_transformer_from_file` (and the cast) on the `_LTX2ComponentLoading`
      mixin, so that rule demanded a declaration from a loader that does implement it.

    So this follows the dispatch: start at the `_load_model` that would run, walk the `self.<helper>()`
    calls each method makes into whichever class in this loader's own MRO defines them, and follow
    `super().<name>()` past the definition it overrides. Per method, not per class, so an unrelated
    method on the same mixin cannot vouch for this one.
    """
    own = [klass for klass in loader_cls.__mro__ if klass not in _GENERIC]
    pending: list[tuple[str, int]] = [("_load_model", 0)]
    seen: set[tuple[str, int]] = set()

    while pending:
        name, start = pending.pop()
        if (name, start) in seen:
            continue
        seen.add((name, start))

        index = next((i for i in range(start, len(own)) if name in vars(own[i])), None)
        if index is None:
            continue  # defined by the generic loader, or not a method of this loader at all

        source = _method_source(own[index], name)
        calls = _self_calls(source)
        if set(_HELPERS) & calls:
            return True
        pending.extend((called, 0) for called in calls)
        pending.extend((called, index + 1) for called in _super_calls(source))

    return False


def _offered_keys() -> dict[tuple[BaseModelType, ModelType, ModelFormat], type]:
    """The registrations a client could offer the control for, by loader key.

    Quantized formats are left out: they are refused by a rule that holds for every loader, present
    and future, so asking each one to declare it would be asking them to repeat the rule.
    """
    return {
        (base, type_, format_): ModelLoaderRegistry._registry[
            ModelLoaderRegistry._to_registry_key(base, type_, format_)
        ]
        for (base, type_, format_) in declared_fp8_storage()
        if type_ in FP8_STORAGE_MODEL_TYPES and format_ not in QUANTIZED_MODEL_FORMATS
    }


class TestEveryLoaderAnswersForItself:
    def test_each_offered_loader_either_implements_the_cast_or_says_why_not(self) -> None:
        """The cell that keeps this from rotting, in both directions.

        A loader with neither is the original defect: a control that changes nothing. A loader with
        both is a stale marker -- it implemented the cast and the declaration was left behind, so the
        UI goes on hiding a control that now works. Neither is allowed to pass quietly.
        """
        undeclared: list[str] = []
        stale: list[str] = []
        for (base, type_, format_), loader_cls in sorted(_offered_keys().items()):
            declaration = declared_fp8_storage()[(base, type_, format_)]
            key = f"{base.value}/{type_.value}/{format_.value} ({loader_cls.__name__})"
            if _implements_the_cast(loader_cls):
                if declaration is not None:
                    stale.append(f"{key}: declares {type(declaration).__name__} but does reach the cast")
            elif declaration is None:
                undeclared.append(f"{key}: never reaches the cast and declares nothing")

        assert undeclared == [], (
            "These loaders would be offered FP8 Storage and silently ignore it. Either implement the "
            "cast or declare NotApplicable/Unimplemented at the registration:\n  " + "\n  ".join(undeclared)
        )
        assert stale == [], (
            "These declarations are out of date -- the loader implements the cast now, so the "
            "declaration only hides a control that works:\n  " + "\n  ".join(stale)
        )

    def test_the_scan_above_can_see_every_way_a_loader_reaches_the_helpers(self) -> None:
        """The scan looks for `self.<helper>(...)` inside a loader class, so any other route is invisible.

        A module-level function taking the loader as an argument would break it silently: the loader
        would read as inert, and this file would then demand a declaration for a loader that works --
        or, if one were added, hide a working control. Pinning the shape the scan assumes is cheaper
        and more honest than making the scan cleverer.
        """
        stray: list[str] = []
        for path in sorted(_LOADERS_DIR.glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Name) and node.id in _HELPERS:
                    stray.append(f"{path.name}:{node.lineno}: bare reference to {node.id}")
                elif isinstance(node, ast.Attribute) and node.attr in _HELPERS:
                    through_self = isinstance(node.value, ast.Name) and node.value.id == "self"
                    if not through_self:
                        stray.append(f"{path.name}:{node.lineno}: {node.attr} reached other than through `self`")
        assert stray == [], (
            "An fp8 helper is reached other than as `self.<helper>(...)`, which the scan above cannot "
            "attribute to a loader class:\n  " + "\n  ".join(stray)
        )

    def test_the_declarations_are_the_backlog_and_nothing_else(self) -> None:
        """`Unimplemented` is work and `NotApplicable` is a decision, so the split has to stay real.

        Asserted as the current lists rather than as counts: a count going down says nothing about
        which one moved, and the useful failure here is "you implemented one, update this line".
        """
        by_marker: dict[type, list[str]] = {NotApplicable: [], Unimplemented: []}
        for (base, type_, format_), declaration in sorted(declared_fp8_storage().items()):
            if declaration is not None:
                by_marker[type(declaration)].append(f"{base.value}/{type_.value}/{format_.value}")

        assert by_marker[NotApplicable] == [
            "anima/controlnet/checkpoint",  # 8-66 MB adapters
            "ideogram-4/main/diffusers",  # published only as nf4 or fp8
        ]
        assert by_marker[Unimplemented] == [
            "flux/controlnet/checkpoint",
            "flux/controlnet/diffusers",
            "minimax-h3/main/checkpoint",
            "minimax-h3/main/diffusers",
            "z-image/controlnet/checkpoint",
        ]

    def test_every_declaration_gives_a_reason(self) -> None:
        """A marker without a reason is a dead control with better manners."""
        for key, declaration in declared_fp8_storage().items():
            if declaration is not None:
                assert declaration.reason.strip(), key
                assert len(declaration.reason) > 30, f"{key}: {declaration.reason!r} explains nothing"


class TestThePolicyMatchesTheSchemas:
    def test_the_types_that_can_ask_are_exactly_the_types_that_carry_the_setting(self) -> None:
        """`FP8_STORAGE_MODEL_TYPES` is derived from the config schemas, so it has to keep matching them.

        Walking every concrete config class and reading the `default_settings` annotation is the only
        way to notice a new type whose settings carry `fp8_storage` -- it would then offer the control
        with nothing behind it, which is where this whole class of bug came from.

        ControlLoRa is the one type that carries the field and is still excluded: it shares
        `ControlAdapterDefaultSettings` with the control adapters but is patched into a base model, so
        the casting hooks would never fire.
        """
        carries_the_field: set[ModelType] = set()
        for config_cls in Config_Base.CONFIG_CLASSES:
            field = config_cls.model_fields.get("default_settings")
            if field is None:
                continue
            settings_types = [
                annotation
                for annotation in getattr(field.annotation, "__args__", (field.annotation,))
                if isinstance(annotation, type) and hasattr(annotation, "model_fields")
            ]
            if any("fp8_storage" in settings.model_fields for settings in settings_types):
                carries_the_field.add(config_cls.model_fields["type"].default)

        assert carries_the_field - FP8_STORAGE_MODEL_TYPES == {ModelType.ControlLoRa}
        assert FP8_STORAGE_MODEL_TYPES - carries_the_field == set()


class TestTheVerdict:
    def test_a_loader_that_implements_the_cast_is_supported(self) -> None:
        """The positive control. Without it every cell below passes on a verdict that always says no."""
        verdict = fp8_storage_verdict(BaseModelType.Flux, ModelType.Main, ModelFormat.Checkpoint)
        assert verdict.supported is True
        assert verdict.reason is None

    def test_a_declared_loader_is_refused_with_its_own_reason(self) -> None:
        verdict = fp8_storage_verdict(BaseModelType.ZImage, ModelType.ControlNet, ModelFormat.Checkpoint)
        assert verdict.supported is False
        assert verdict.reason is not None and "never cast" in verdict.reason

    def test_a_quantized_format_is_refused_although_its_loader_declares_nothing(self) -> None:
        """The rule the loaders do not have to repeat. `WanGGUFCheckpointModel` declares nothing and
        must still come out unsupported, or every quantized loader would need a marker saying the same
        sentence."""
        assert declared_fp8_storage()[(BaseModelType.Wan, ModelType.Main, ModelFormat.GGUFQuantized)] is None
        verdict = fp8_storage_verdict(BaseModelType.Wan, ModelType.Main, ModelFormat.GGUFQuantized)
        assert verdict.supported is False
        assert verdict.reason is not None and "packed quantized payload" in verdict.reason

    @pytest.mark.parametrize(
        "type_",
        [ModelType.VAE, ModelType.LoRA, ModelType.ControlLoRa, ModelType.IPAdapter],
        ids=lambda t: t.value,
    )
    def test_a_type_the_setting_cannot_reach_is_refused_before_anything_else_is_asked(self, type_: ModelType) -> None:
        verdict = fp8_storage_verdict(BaseModelType.Flux, type_, ModelFormat.Diffusers)
        assert verdict.supported is False
        # The reason names the type and stops there. Each of these is excluded on different grounds --
        # decode quality for a VAE, patched-in weights for the LoRAs, no such setting for the rest --
        # and one sentence covering all four would have to be untrue of most of them.
        assert verdict.reason == f"FP8 Storage is not offered for {type_.value} models."

    def test_a_wildcard_registration_answers_for_every_base_it_serves(self) -> None:
        """`get_implementation` falls back to the `Any` base, so this has to as well.

        Every T2I adapter is loaded by `GenericDiffusersLoader` under `any/t2i_adapter/diffusers`, and no
        record carries `any` as its own base. Looking up the exact key alone answers "no loader
        registered" for all of them, which would quietly withdraw a control that works.
        """
        assert (BaseModelType.StableDiffusion1, ModelType.T2IAdapter, ModelFormat.Diffusers) not in (
            declared_fp8_storage()
        )
        assert (BaseModelType.Any, ModelType.T2IAdapter, ModelFormat.Diffusers) in declared_fp8_storage()

        for base in (BaseModelType.StableDiffusion1, BaseModelType.StableDiffusionXL):
            assert fp8_storage_verdict(base, ModelType.T2IAdapter, ModelFormat.Diffusers).supported is True

    def test_an_exact_registration_wins_over_the_wildcard(self) -> None:
        """Same precedence as the registry's own lookup, including when the exact one is the stricter.

        No architecture registers its own T2I adapter loader today, so this is asserted against the
        resolution rather than against a live pair: a declared key must not be overruled by a permissive
        wildcard sitting behind it.
        """
        key = (BaseModelType.Flux, ModelType.T2IAdapter, ModelFormat.Diffusers)
        # Asserted before writing: the day a FLUX-specific T2I adapter loader registers, this cell must
        # fail rather than delete that loader's real declaration on its way out.
        assert key not in declared_fp8_storage(), "this key is registered now; pin the precedence against it instead"

        declare_fp8_storage(*key, Unimplemented("registered by this test only, to pin the precedence"))
        try:
            assert fp8_storage_verdict(*key).supported is False
            assert (
                fp8_storage_verdict(
                    BaseModelType.StableDiffusion1, ModelType.T2IAdapter, ModelFormat.Diffusers
                ).supported
                is True
            )
        finally:
            _undeclare_for_test(key)

    def test_a_key_with_no_loader_is_refused(self) -> None:
        """Nothing can load it, so nothing can honour the setting either -- and the client's lookup
        misses for exactly these, which is why absence has to mean no rather than yes."""
        key = (BaseModelType.StableDiffusion1, ModelType.Main, ModelFormat.BnbQuantizednf4b)
        assert key not in declared_fp8_storage()
        verdict = fp8_storage_verdict(*key)
        assert verdict.supported is False


class TestTheServedTable:
    def test_flux_main_and_flux_controlnet_disagree(self) -> None:
        """The reason the table is keyed `(base, type, format)` and not by architecture.

        Both rows are `flux`. A per-architecture field -- the shape this was first designed as -- can
        hold only one answer for the two, and would have to be wrong about one of them.
        """
        rows = {(row.base, row.type, row.format): row.supported for row in fp8_storage_support()}
        assert rows[(BaseModelType.Flux, ModelType.Main, ModelFormat.Checkpoint)] is True
        assert rows[(BaseModelType.Flux, ModelType.ControlNet, ModelFormat.Checkpoint)] is False

    def test_the_format_changes_the_answer_for_one_architecture(self) -> None:
        """The other axis a coarser key loses: same base, same type, two formats, two answers."""
        rows = {(row.base, row.type, row.format): row.supported for row in fp8_storage_support()}
        assert rows[(BaseModelType.Wan, ModelType.Main, ModelFormat.Checkpoint)] is True
        assert rows[(BaseModelType.Wan, ModelType.Main, ModelFormat.GGUFQuantized)] is False

    def test_every_loader_of_those_types_gets_exactly_one_row(self) -> None:
        """Counted against the registry rather than rebuilt from the table's own comprehension, which
        would only restate it. A client builds a map from this, so a duplicate would pick a winner in
        silence and a missing row would read as unsupported."""
        rows = fp8_storage_support()
        keys = [(row.base, row.type, row.format) for row in rows]

        assert len(set(keys)) == len(keys)
        assert len(keys) == sum(
            1 for _base, type_, _format in declared_fp8_storage() if type_ in FP8_STORAGE_MODEL_TYPES
        )
