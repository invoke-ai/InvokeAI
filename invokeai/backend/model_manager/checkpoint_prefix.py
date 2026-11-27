"""The wrapper prefix a single-file checkpoint's keys carry, as one object.

Redistributors wrap a model's keys in a namespace of their own — ``model.diffusion_model.`` for the
ComfyUI bundle layout, ``diffusion_model.`` for the bare one, ``net.`` for Anima — and every loader
that accepts such a file has to undo that before anything reads a key. Nine near-identical copies of
that step had grown across six modules, six of them inline in a loader method, differing only in
which prefixes they listed and (for Anima alone) in whether keys outside the prefix are kept.

A loader whose list is short by one entry does not fail quietly, but it fails confusingly: every key
keeps its wrapper, so nothing matches, and the load dies on the *missing* keys rather than on the
prefix — ``load_state_dict_ignoring_extras`` logs the unexpected ones and raises for the absent ones
(``backend/util/state_dict_loading.py``). It is a defect per loader rather than one to fix once.

Identification reads the same prefixes to decide *which* loader a file goes to
(``configs/main.py``), so this module deliberately imports nothing: a cycle or a torch import here
would put the whole loading stack into identification-time code. ``backend/pid/state_dict_utils.py``
is kept dependency-free for the same reason.

Two rules this deliberately does *not* unify:

- **Per-key stripping.** The Mistral encoder strips ``text_encoder.``/``language_model.`` from each
  key independently rather than detecting one prefix for the whole file, because its redistributions
  mix wrapped and unwrapped keys. Identification strips per key too, for the same reason: it is
  looking for evidence in individual names, not normalising a file. Both share the list below;
  neither can share :meth:`CheckpointPrefix.detect`.
- **The header hints.** ``strip_layer_path_prefix`` strips any known prefix from a layer name named
  in ``_quantization_metadata``, whatever the state dict carried. Binding it to the prefix detected
  here would be stricter and would close a drift — but it would also lose the hint on a file whose
  header names a namespace its tensors do not use, which works today. Neither shape is observed on
  any checkpoint measured (0 of the 7 local files carrying header metadata name a layer their
  tensors do not), so the forgiving rule is kept: it cannot lose a hint that currently matches.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

#: What the ComfyUI single-file conventions wrap a transformer in. ``net.`` is deliberately *not*
#: here: Anima passes it in its own list, while for a PiD checkpoint ``net.`` is the model's real
#: root module rather than a wrapper (``backend/pid/state_dict_utils.py`` strips it with a deny-list
#: of its own). A shared default carrying it would teach every loader to cut into a namespace that
#: is load-bearing elsewhere.
#:
#: These two are mutually exclusive as prefixes — ``model.diffusion_model.x`` does *not* start with
#: ``diffusion_model.`` — so their order here is not load-bearing. It would become load-bearing the
#: moment an entry is added that is a prefix of another, which is why :meth:`CheckpointPrefix.detect`
#: takes the first match in *list* order rather than the longest.
COMFYUI_KEY_PREFIXES = ("model.diffusion_model.", "diffusion_model.")

T = TypeVar("T")


@dataclass(frozen=True)
class CheckpointPrefix:
    """The one prefix a checkpoint's keys carry, or the empty string for a bare checkpoint.

    Built from the dict, then applied to it. It reports nothing beyond :attr:`prefix` because there
    is nothing else to report: for a prefix strip the whole source-to-destination mapping follows
    from that one string. A transform that renames per key does have to return its mapping, so that
    a layer's side channel can be moved to wherever its weight went.
    """

    prefix: str

    @classmethod
    def detect(cls, sd: Mapping[Any, Any], prefixes: Iterable[str] = COMFYUI_KEY_PREFIXES) -> "CheckpointPrefix":
        """The first of ``prefixes`` that any key starts with, in the order given."""
        for prefix in prefixes:
            if any(isinstance(key, str) and key.startswith(prefix) for key in sd):
                return cls(prefix)
        return cls("")

    def strip(self, sd: dict[Any, T], *, drop_foreign: bool = False) -> dict[Any, T]:
        """``sd`` re-keyed into the model's own namespace.

        A bare checkpoint is returned as it came, not copied: loaders drop their reference to the
        dict to let the weights go before a cast, and an extra copy here would keep them alive.

        ``drop_foreign`` discards what was *not* under the prefix, which is what a loader reading an
        all-in-one export wants: the bundled VAE and text encoder sit beside the transformer under
        namespaces of their own, and they are not this model's. It applies only once a prefix has
        been detected — a bare checkpoint has nothing to tell foreign keys apart by, and dropping
        them there would hide a stray key from the unexpected-key checks that exist to report it.
        """
        if not self.prefix:
            return sd
        cut = len(self.prefix)
        if drop_foreign:
            return {
                key[cut:]: value for key, value in sd.items() if isinstance(key, str) and key.startswith(self.prefix)
            }
        return {
            (key[cut:] if isinstance(key, str) and key.startswith(self.prefix) else key): value
            for key, value in sd.items()
        }
