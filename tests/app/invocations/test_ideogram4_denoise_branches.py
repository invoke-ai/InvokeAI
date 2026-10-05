"""How `Ideogram4DenoiseInvocation` resolves its two transformer branches.

Ideogram 4 guides one branch against the other and runs both at every step. A diffusers pipeline
delivers them as one cache entity; Comfy-Org's single files are two models, and both must be locked
for the whole loop -- releasing one between steps would make the cache stream it back for the next.
The mismatches are rejected rather than tolerated: either one would otherwise surface deep in the
loop, as a missing attribute or as a branch guiding against itself.
"""

import logging
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace

import pytest
import torch

from invokeai.app.invocations.fields import Ideogram4ConditioningField
from invokeai.app.invocations.ideogram4.ideogram4_denoise import Ideogram4DenoiseInvocation
from invokeai.app.invocations.model import ModelIdentifierField, TransformerField
from invokeai.backend.ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from invokeai.backend.ideogram4.transformer_pair import Ideogram4TransformerPair
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.backend.quantization.int8_convrot import Int8ConvrotLinear
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import (
    ConditioningFieldData,
    Ideogram4ConditioningInfo,
)
from invokeai.backend.util.devices import TorchDevice

TINY = Ideogram4Config(
    emb_dim=64,
    num_layers=1,
    num_heads=2,
    intermediate_size=128,
    adanln_dim=16,
    in_channels=8,
    llm_features_dim=32,
    mrope_section=(4, 2, 2),
)


def _field(key: str) -> TransformerField:
    return TransformerField(
        transformer=ModelIdentifierField(
            key=key, hash=f"hash-{key}", name=key, base=BaseModelType.Ideogram4, type=ModelType.Main
        ),
        loras=[],
    )


class _Residency:
    """What the cap reads off a branch: where it lives, how big it is, how much of it is resident."""

    def __init__(
        self,
        device: str = "cpu",
        weight_bytes: int = 0,
        resident_weight_bytes: int = 0,
        supports_partial_loading: bool = True,
    ) -> None:
        self.device = device
        self.weight_bytes = weight_bytes
        self.resident_weight_bytes = resident_weight_bytes
        self.supports_partial_loading = supports_partial_loading
        self.unloaded: list[tuple[int, bool]] = []


class _LoadedModel:
    """The slice of `LoadedModel` this method uses, plus a record of how it was entered and left.

    `model` is the real attribute name: the node reads it before locking, to see whether the branch
    is an int8 build whose per-forward dequantization needs headroom of its own.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        calls: list[tuple[str, int]],
        released: list[str],
        key: str,
        residency: "_Residency | None" = None,
    ) -> None:
        self.model = model
        self._model = model
        self._calls = calls
        self._released = released
        self._key = key
        # Default to a CPU branch, which is the shape every reservation test wants: the cap has
        # nothing to stream there and returns before touching any of the fields below.
        residency = residency or _Residency(device="cpu")
        self.compute_device = torch.device(residency.device)
        self.supports_partial_loading = residency.supports_partial_loading
        self.weight_bytes = residency.weight_bytes
        self.resident_weight_bytes = residency.resident_weight_bytes
        self.unloaded: list[tuple[int, bool]] = residency.unloaded

    @contextmanager
    def model_on_device(self, working_mem_bytes: int = 0):
        self._calls.append((self._key, working_mem_bytes))
        try:
            yield (None, self._model)
        finally:
            self._released.append(self._key)

    def unload_from_vram(self, vram_bytes_to_free: int, keep_required_weights_in_vram: bool = False) -> int:
        self.unloaded.append((vram_bytes_to_free, keep_required_weights_in_vram))
        return vram_bytes_to_free


def _context(
    models: dict[str, torch.nn.Module],
    released: list[str],
    calls: list[tuple[str, int]] | None = None,
    loaded: list[str] | None = None,
    residency: "dict[str, _Residency] | None" = None,
    working_mem_gb: float = 3.0,
    max_cache_vram_gb: float | None = None,
) -> SimpleNamespace:
    """`calls` records the *locks* and their reservations; `loaded` records the RAM-level loads.

    They are separate because the node's ordering distinguishes them: it reads both branches before
    locking either, and refuses a mis-wired graph before reading the second.
    """
    recorded = calls if calls is not None else []
    reads = loaded if loaded is not None else []

    def load(identifier: ModelIdentifierField) -> _LoadedModel:
        reads.append(identifier.key)
        return _LoadedModel(
            models[identifier.key],
            recorded,
            released,
            identifier.key,
            (residency or {}).get(identifier.key),
        )

    return SimpleNamespace(
        models=SimpleNamespace(load=load),
        logger=logging.getLogger("test"),
        # The node reads `device_working_mem_gb` because the cache raises any smaller reservation to
        # it; the default here is the repository default.
        config=SimpleNamespace(
            get=lambda: SimpleNamespace(device_working_mem_gb=working_mem_gb, max_cache_vram_gb=max_cache_vram_gb)
        ),
    )


def _invocation(**fields) -> Ideogram4DenoiseInvocation:
    return Ideogram4DenoiseInvocation.model_construct(unconditional_transformer=None, **fields)


def test_a_bundled_pair_supplies_both_branches_from_one_model() -> None:
    pair = Ideogram4TransformerPair(conditional=Ideogram4Transformer(TINY), unconditional=Ideogram4Transformer(TINY))
    released: list[str] = []
    invocation = _invocation(transformer=_field("pipeline"))

    with ExitStack() as stack:
        conditional, unconditional = invocation._load_branches(_context({"pipeline": pair}, released), stack, 0)

        assert conditional is pair.conditional
        assert unconditional is pair.unconditional
        assert released == []

    assert released == ["pipeline"]


def test_two_single_files_are_both_held_for_the_whole_loop() -> None:
    models = {"cond": Ideogram4Transformer(TINY), "uncond": Ideogram4Transformer(TINY)}
    released: list[str] = []
    invocation = _invocation(transformer=_field("cond"))
    invocation.unconditional_transformer = _field("uncond")

    loaded: list[str] = []
    calls: list[tuple[str, int]] = []
    with ExitStack() as stack:
        conditional, unconditional = invocation._load_branches(_context(models, released, calls, loaded), stack, 0)

        assert conditional is models["cond"]
        assert unconditional is models["uncond"]
        # Neither is handed back while the other is still being loaded.
        assert released == []

    assert sorted(released) == ["cond", "uncond"]
    # Both branches are read before either is locked, which is what lets the reservation below be
    # the maximum of the two rather than each branch's own.
    assert loaded == ["cond", "uncond"]
    assert [key for key, _ in calls] == ["cond", "uncond"]


def test_a_lone_single_file_is_refused() -> None:
    # Without this the loop would ask a bare transformer for `.conditional` and die on an
    # AttributeError with no hint about what the user should have connected.
    released: list[str] = []
    invocation = _invocation(transformer=_field("cond"))

    with pytest.raises(ValueError, match="only one branch"), ExitStack() as stack:
        invocation._load_branches(_context({"cond": Ideogram4Transformer(TINY)}, released), stack, 0)


def test_a_bundled_pair_with_a_second_branch_connected_is_refused() -> None:
    pair = Ideogram4TransformerPair(conditional=Ideogram4Transformer(TINY), unconditional=Ideogram4Transformer(TINY))
    released: list[str] = []
    invocation = _invocation(transformer=_field("pipeline"))
    invocation.unconditional_transformer = _field("uncond")

    loaded: list[str] = []
    with pytest.raises(ValueError, match="already carries both"), ExitStack() as stack:
        invocation._load_branches(
            _context({"pipeline": pair, "uncond": Ideogram4Transformer(TINY)}, released, loaded=loaded), stack, 0
        )

    # The refusal comes before the second branch is read: a mis-wired graph costs an error, not a
    # ~9 GiB load of a model the node is about to reject. The node reads both branches before it
    # locks either, so this is the ordering that has to be pinned, not the locking.
    assert loaded == ["pipeline"]


def _int8_branch() -> Ideogram4Transformer:
    """A tiny branch whose largest linear is stored int8, as the `int8_convrot` build's are.

    In bfloat16 because that is what the loader produces, and the transient is two weight-sized
    tensors *in the compute dtype* -- a float32 model would quietly double the expected number.
    """
    model = Ideogram4Transformer(TINY).to(torch.bfloat16)
    linear = model.layers[0].feed_forward.w1
    model.layers[0].feed_forward.w1 = Int8ConvrotLinear(
        weight=torch.zeros(linear.out_features, linear.in_features, dtype=torch.int8),
        weight_scale=torch.ones(linear.out_features, 1),
        convrot=False,
    )
    return model


def test_an_int8_branch_adds_its_dequantization_headroom() -> None:
    """`Int8ConvrotLinear` materializes the dequantized weight inside `forward`.

    That peak is not part of the model's resident size, so it has to be reserved — and it is
    measured from the model, since a bf16 or fp8 branch needs none of it.
    """
    dense = Ideogram4Transformer(TINY).to(torch.bfloat16)
    int8 = _int8_branch()
    largest = int8.layers[0].feed_forward.w1

    assert Ideogram4DenoiseInvocation._dequant_transient(dense) == 0
    assert Ideogram4DenoiseInvocation._dequant_transient(int8) == (
        2 * largest.in_features * largest.out_features * torch.bfloat16.itemsize
    )


@pytest.mark.parametrize("int8_branch", ["cond", "uncond"])
def test_a_mixed_pair_reserves_for_the_hungrier_branch(int8_branch: str) -> None:
    """The cache keeps the *last* lock's reservation, not the sum.

    Free VRAM is recomputed as `capacity - working_mem - in_use` at every lock, so a branch that
    reserves less hands back exactly the headroom the other one still needs. Both orders are pinned
    because both occur: the loader node allows an int8 branch guided against an fp8 one in either
    slot, and reserving only for the first branch would leave one of the two orders green.
    """
    models = {"cond": Ideogram4Transformer(TINY).to(torch.bfloat16), "uncond": Ideogram4Transformer(TINY)}
    models[int8_branch] = _int8_branch()
    calls: list[tuple[str, int]] = []
    invocation = _invocation(transformer=_field("cond"))
    invocation.unconditional_transformer = _field("uncond")
    transient = Ideogram4DenoiseInvocation._dequant_transient(models[int8_branch])
    assert transient > 0

    with ExitStack() as stack:
        invocation._load_branches(_context(models, [], calls), stack, 4 * 1024**3)

    assert calls == [("cond", 4 * 1024**3 + transient), ("uncond", 4 * 1024**3 + transient)]


def test_both_branches_reserve_the_same_activation_headroom() -> None:
    """A reservation given to only the first branch is spent by the second.

    The cache decides per model how much of it fits in what is left, so the second `model_on_device`
    would fill the headroom the first one set aside — which is how the pair ended up occupying every
    byte of a 24 GB card and copying weights instead of computing.
    """
    models = {"cond": Ideogram4Transformer(TINY), "uncond": Ideogram4Transformer(TINY)}
    calls: list[tuple[str, int]] = []
    invocation = _invocation(transformer=_field("cond"))
    invocation.unconditional_transformer = _field("uncond")

    with ExitStack() as stack:
        invocation._load_branches(_context(models, [], calls), stack, 4 * 1024**3)

    assert calls == [("cond", 4 * 1024**3), ("uncond", 4 * 1024**3)]


def test_the_headroom_estimate_grows_with_the_image() -> None:
    """It is an estimate, so what is pinned is the shape: it scales with the token count and is
    never zero. A flat value would under-reserve at 2048px and over-reserve at 512px."""
    small = _invocation(transformer=_field("cond"))
    small.width, small.height = 512, 512
    large = _invocation(transformer=_field("cond"))
    large.width, large.height = 2048, 2048

    assert small._estimate_working_memory(64) > 1024**3
    # Sixteen times the image tokens, so the token-proportional part must dominate the fixed base.
    assert large._estimate_working_memory(64) > 4 * small._estimate_working_memory(64)


TEXT_TOKENS = 7


class _StopAfterLoad(Exception):
    """Ends `invoke` where this test's interest ends."""


def test_invoke_passes_its_own_estimate_into_the_load(monkeypatch) -> None:
    """The wiring, not the estimate.

    Every other test here drives `_load_branches` directly, so a call in `invoke` that forgot the
    argument -- or passed a constant -- would leave them all green while the cache went back to
    filling VRAM with weights and leaving none for the activations.
    """
    reserved: list[int] = []
    invocation = _invocation(
        transformer=_field("cond"),
        positive_conditioning=Ideogram4ConditioningField(conditioning_name="cond-1"),
        sampler_preset="V4_TURBO_12",
        steps=None,
        guidance_scale=None,
        mu=None,
        width=1024,
        height=1024,
        seed=0,
    )

    def load_branches(_context, _stack, working_mem_bytes: int):
        reserved.append(working_mem_bytes)
        raise _StopAfterLoad

    monkeypatch.setattr(invocation, "_load_branches", load_branches)

    conditioning = ConditioningFieldData(
        conditionings=[Ideogram4ConditioningInfo(prompt_embeds=torch.zeros(TEXT_TOKENS, 32))]
    )
    context = SimpleNamespace(
        conditioning=SimpleNamespace(load=lambda _name: conditioning),
        util=SimpleNamespace(signal_progress=lambda *_args, **_kwargs: None),
    )

    with pytest.raises(_StopAfterLoad):
        invocation.invoke(context)

    assert reserved == [invocation._estimate_working_memory(TEXT_TOKENS)]


GIB = 1024**3
# The shipped fp8 branch is 8891.78 MB. The reservation at 1024x1024 is about 5 GiB, which is also
# the cache's configured floor on the machine these numbers come from.
BRANCH_BYTES = 8891 * 1024**2
RESERVATION = 5 * GIB
WORKING_MEM_GB = 5.0


def _drive(
    monkeypatch,
    total_vram: int,
    *,
    cond: "_Residency | None" = None,
    uncond: "_Residency | None" = None,
    working_mem_gb: float = WORKING_MEM_GB,
    reservation: int = RESERVATION,
    free_vram: int | None = None,
    max_cache_vram_gb: float | None = None,
) -> tuple["_Residency", "_Residency", list[tuple[str, int]], list[str]]:
    """Run `_load_branches` with both branches on a fake CUDA device of `total_vram` bytes.

    `mem_get_info` is stubbed rather than skipped off CUDA hardware: what is under test is the
    node's arithmetic against a device's capacity, which is worth pinning wherever the suite runs.
    `empty_cache` is stubbed too -- the real one is a process-global free, which a unit test has no
    business issuing -- and its calls are recorded, because without it the freed blocks stay in
    torch's reserve and the room the node just made is invisible to anything outside the process.
    """
    # Free defaults to the device minus a desktop's worth and the first branch, which is the state
    # the second lock actually meets.
    free = total_vram - (3 * GIB) // 2 - BRANCH_BYTES if free_vram is None else free_vram
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (free, total_vram))
    emptied: list[str] = []
    monkeypatch.setattr(TorchDevice, "empty_cache", staticmethod(lambda: emptied.append("empty_cache")))

    cond = cond or _Residency(device="cuda", weight_bytes=BRANCH_BYTES, resident_weight_bytes=BRANCH_BYTES)
    uncond = uncond or _Residency(device="cuda", weight_bytes=BRANCH_BYTES, resident_weight_bytes=BRANCH_BYTES)
    models = {"cond": Ideogram4Transformer(TINY), "uncond": Ideogram4Transformer(TINY)}
    invocation = _invocation(transformer=_field("cond"))
    invocation.unconditional_transformer = _field("uncond")
    calls: list[tuple[str, int]] = []

    with ExitStack() as stack:
        invocation._load_branches(
            _context(
                models,
                [],
                calls,
                residency={"cond": cond, "uncond": uncond},
                working_mem_gb=working_mem_gb,
                max_cache_vram_gb=max_cache_vram_gb,
            ),
            stack,
            reservation,
        )
    return cond, uncond, calls, emptied


def _target_for(total_vram: int, other_branch: int = BRANCH_BYTES, reservation: int = RESERVATION) -> int:
    """What is left for the second branch once the system, the reservation and the other branch are
    subtracted. Written out here rather than imported so a change of policy fails these cells."""
    return total_vram - 3 * GIB - reservation - other_branch


def test_the_second_branch_is_held_to_what_the_card_has_left(monkeypatch) -> None:
    """The target is the deficit, not a fixed number.

    Measured on the real models before this existed: both branches settled at 100% and 96.8%
    residency and the card peaked at 23288 of 24564 MiB -- the whole device, on a machine whose GPU
    also drives the display. With the target derived, the same generation settled branch B at 84%
    and peaked at 22117 MiB; raising `device_working_mem_gb` to 8 moved it to 50% and 19009 MiB.
    """
    _cond, uncond, _calls, emptied = _drive(monkeypatch, total_vram=24 * GIB)

    target = _target_for(24 * GIB)
    assert 2 * GIB < target < BRANCH_BYTES, "the fixture must exercise a derived target, not the floor"
    # The tensors a streamed forward cannot fetch per layer stay behind.
    assert uncond.unloaded == [(BRANCH_BYTES - target, True)]
    # Without returning the blocks to the driver the room exists only inside this process.
    assert emptied == ["empty_cache"]


def test_the_lock_is_told_the_target_so_the_weights_are_never_loaded_twice(monkeypatch) -> None:
    """The reservation is the lever; the unload is only a correction.

    Letting the cache fill the branch and pushing it back afterwards moves ~6 GiB onto the device
    and straight off again on every invocation. The second lock therefore gets a reservation that
    leaves exactly the target free, which is what Wan's expert swapper does.
    """
    _cond, _uncond, calls, _emptied = _drive(monkeypatch, total_vram=24 * GIB)

    (first_key, first_reservation), (second_key, second_reservation) = calls
    assert (first_key, second_key) == ("cond", "uncond")
    # The conditional branch is not capped: it is locked first and every step runs it.
    assert first_reservation == RESERVATION
    # `_get_vram_available` is `free - working_mem`, so this leaves the target and nothing more.
    free = 24 * GIB - (3 * GIB) // 2 - BRANCH_BYTES
    assert second_reservation == free - _target_for(24 * GIB)
    # And it is genuinely an inflation, not the node's own number arriving unchanged.
    assert second_reservation > RESERVATION


def test_a_lopsided_pair_is_measured_rather_than_doubled(monkeypatch) -> None:
    """The two branches can be different builds, so neither one's size stands in for the pair.

    The loader accepts an int8 branch guiding against an fp8 one, and an fp8 file that is expanded
    rather than kept quantized is about twice the size of either. Doubling the second branch here
    would compute 17.4 GiB for a pair that is really 25.7 GiB, clear the gate on a 32 GiB card and
    leave the display with nothing.
    """
    expanded = 17 * GIB
    cond = _Residency(device="cuda", weight_bytes=expanded, resident_weight_bytes=expanded)
    _cond, uncond, _calls, _emptied = _drive(monkeypatch, total_vram=32 * GIB, cond=cond)

    assert uncond.unloaded, "a 17 GiB + 8.7 GiB pair does not fit a 32 GiB card"
    assert uncond.unloaded == [(BRANCH_BYTES - _target_for(32 * GIB, other_branch=expanded), True)]


def test_the_gate_uses_the_reservation_the_cache_will_actually_apply(monkeypatch) -> None:
    """`_get_vram_available` raises any reservation below `device_working_mem_gb`.

    At 512px the node's own estimate is far under that floor, so asking the gate with it would
    clear a card the cache then fills anyway -- both branches resident and under 2 GiB left for the
    display, which is the condition this whole path exists to detect.
    """
    # 24 - 3 - 2 - 8.68 leaves room for the second branch; 24 - 3 - 5 - 8.68 does not.
    _cond, uncond, _calls, _emptied = _drive(monkeypatch, total_vram=24 * GIB, reservation=2 * GIB, working_mem_gb=5.0)

    assert uncond.unloaded, "the gate must use the 5 GiB floor, not the 2 GiB estimate"


def test_a_card_with_room_for_the_pair_keeps_both_resident(monkeypatch) -> None:
    """The cap answers a device that is too small, not a policy about Ideogram: without this, a
    48 GB card would pay the streaming cost for headroom it already has."""
    _cond, uncond, calls, emptied = _drive(monkeypatch, total_vram=48 * GIB)

    assert uncond.unloaded == []
    assert emptied == []
    # And both locks are given the node's own reservation, unmodified.
    assert calls == [("cond", RESERVATION), ("uncond", RESERVATION)]


def test_a_card_just_above_the_threshold_is_left_alone(monkeypatch) -> None:
    """Brackets the system's share from the other side.

    The 48 GiB cell alone would stay green if the reserve were raised to any value below ~26 GiB,
    which would start capping 32 and 40 GB cards that have room to spare.
    """
    # 2 x 8.68 + 5 + 3 = 25.4 GiB, so 26 GiB clears it with little to spare.
    _cond, uncond, _calls, _emptied = _drive(monkeypatch, total_vram=26 * GIB)

    assert uncond.unloaded == []


def test_a_branch_already_below_the_target_is_not_touched(monkeypatch) -> None:
    """A second invocation finds the branch as the first one left it. Trimming again would ask the
    cache to free bytes that are not there, and an unconditional call would log a 0 GiB stream."""
    low = _Residency(device="cuda", weight_bytes=BRANCH_BYTES, resident_weight_bytes=1 * GIB)
    _cond, uncond, _calls, emptied = _drive(monkeypatch, total_vram=24 * GIB, uncond=low)

    assert uncond.unloaded == []
    assert emptied == []


def test_the_size_and_the_residency_are_not_interchangeable(monkeypatch) -> None:
    """The gate asks how big the pair is; the trim asks how much of it is on the device.

    A second generation in a session finds the branch as the first one left it -- partly resident.
    Reading residency where size belongs would then under-read the pair, decide the card has room
    and hand the second lock an uninflated reservation, so the branch would fill up again and the
    feature would silently stop working exactly when it matters. Pinned on the reservation rather
    than on the unload, because a branch already under the target has nothing to trim.
    """
    half = _Residency(device="cuda", weight_bytes=BRANCH_BYTES, resident_weight_bytes=2 * GIB)
    _cond, uncond, calls, _emptied = _drive(monkeypatch, total_vram=24 * GIB, uncond=half)

    free = 24 * GIB - (3 * GIB) // 2 - BRANCH_BYTES
    assert calls[1] == ("uncond", free - _target_for(24 * GIB))
    # Nothing to trim: the cache already holds less than the branch is allowed to keep.
    assert uncond.unloaded == []


def test_a_branch_that_cannot_stream_is_left_alone(monkeypatch) -> None:
    """With partial loading off the cache holds the model whole, so a partial unload cannot help:
    asking for one drops every weight and makes the loop rebuild it from RAM each step."""
    whole = _Residency(
        device="cuda",
        weight_bytes=BRANCH_BYTES,
        resident_weight_bytes=BRANCH_BYTES,
        supports_partial_loading=False,
    )
    _cond, uncond, calls, _emptied = _drive(monkeypatch, total_vram=24 * GIB, uncond=whole)

    assert uncond.unloaded == []
    # And no inflated reservation either -- it would only starve the branch the cache must load whole.
    assert calls == [("cond", RESERVATION), ("uncond", RESERVATION)]


def test_the_bundled_pair_is_never_capped(monkeypatch) -> None:
    """A diffusers pipeline is one cache record holding *both* branches, and the node returns before
    the cap. Generalizing the cap onto that record would read one branch's size as the pair's, fire
    on almost any card, and then stream both branches -- worse than the problem being fixed."""
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (12 * GIB, 24 * GIB))
    pair_record = _Residency(device="cuda", weight_bytes=2 * BRANCH_BYTES, resident_weight_bytes=2 * BRANCH_BYTES)
    pair = Ideogram4TransformerPair(conditional=Ideogram4Transformer(TINY), unconditional=Ideogram4Transformer(TINY))
    invocation = _invocation(transformer=_field("pipeline"))

    with ExitStack() as stack:
        invocation._load_branches(
            _context({"pipeline": pair}, [], residency={"pipeline": pair_record}), stack, RESERVATION
        )

    assert pair_record.unloaded == []


def test_a_card_short_by_more_than_a_branch_keeps_a_working_set(monkeypatch) -> None:
    """Below roughly 19 GiB the derived target goes negative and the floor becomes the policy.

    Without the clamp both consumers run on a negative number: the second lock is handed a
    reservation larger than the whole free pool, so the cache loads almost none of the branch, and
    the correction then asks it to free more than is resident. A 12-16 GB card would stream every
    layer of every step instead of keeping a working set -- slow turns into unusable, which is
    exactly what the floor exists to prevent.
    """
    assert _target_for(16 * GIB) < 0, "the fixture must reach the floor, not a derived target"
    _cond, uncond, calls, _emptied = _drive(monkeypatch, total_vram=16 * GIB)

    assert uncond.unloaded == [(BRANCH_BYTES - 2 * GIB, True)]
    free = 16 * GIB - (3 * GIB) // 2 - BRANCH_BYTES
    assert calls[1] == ("uncond", max(RESERVATION, free - 2 * GIB))


def test_an_explicitly_capped_cache_is_left_to_its_own_budget(monkeypatch) -> None:
    """With `max_cache_vram_gb` set the cache budgets from that cap, not from the device.

    A target derived from physical memory then describes a machine the cache is not using, and the
    inflated reservation would come off the cap a second time -- on a 24 GiB card capped at 16 that
    pushes the branch under the 2 GiB floor, in the name of freeing VRAM the cache was already
    forbidden to take. Such an install has already chosen its residency by another route.
    """
    _cond, uncond, calls, emptied = _drive(monkeypatch, total_vram=24 * GIB, max_cache_vram_gb=16.0)

    assert uncond.unloaded == []
    assert emptied == []
    assert calls == [("cond", RESERVATION), ("uncond", RESERVATION)]
