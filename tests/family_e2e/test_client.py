import pytest

from scripts.family_e2e.client import Event, build_timing, per_step_seconds
from scripts.family_e2e.graphs import prepare_batch

T0 = 1000.0


def test_step_time_survives_throttled_frames() -> None:
    # 8 steps of 0.5 s; only steps 1, 3, 6 and 8 sent a frame.
    frames = [(T0 + 0.5 * k, k / 8) for k in (1, 3, 6, 8)]
    assert per_step_seconds(frames) == pytest.approx(0.5)


def test_step_time_counts_an_initial_zero_frame_as_step_zero() -> None:
    # SD-style loops send a frame before the first step; img2img runs fewer steps than requested (here 6).
    frames = [(T0, 0.0), (T0 + 0.3 * 2, 2 / 6), (T0 + 0.3 * 5, 5 / 6), (T0 + 0.3 * 6, 1.0)]
    assert per_step_seconds(frames) == pytest.approx(0.3)


def test_frames_on_even_steps_only_need_the_expected_step_count() -> None:
    # 8 steps of 50 ms: the 0.1 s throttle let only every other step's frame through.
    frames = [(T0 + 0.05 * k, k / 8) for k in (0, 2, 4, 6, 8)]
    assert per_step_seconds(frames, expected_steps=8) == pytest.approx(0.05)
    # Inferred, these frames read as 4 steps of 0.1 s; a step that fast cannot rule that out.
    assert per_step_seconds(frames) is None


def test_expected_step_count_that_does_not_fit_falls_back_to_inference() -> None:
    # The node ran 6 steps although the case asked for 8 (e.g. a preset-driven family); 0.5 s per step.
    frames = [(T0 + 0.5 * k, k / 6) for k in (1, 2, 5, 6)]
    assert per_step_seconds(frames, expected_steps=8) == pytest.approx(0.5)


def test_step_time_needs_two_distinct_steps() -> None:
    assert per_step_seconds([(T0, 1.0)]) is None
    assert per_step_seconds([(T0, 1.0), (T0 + 1, 1.0)]) is None


def _invocation(name: str, t: float, node: str, item_id: int = 7, **extra: object) -> Event:
    data = {"item_id": item_id, "invocation": {"id": f"{node}-p", "type": f"{node}_type"}, "invocation_source_id": node}
    return Event(name, t, data | extra)


def test_timing_is_relative_to_enqueue_and_ignores_other_items() -> None:
    load = {"config": {"key": "m"}, "submodel_type": "transformer"}
    events = [
        # The socket client handles events on separate threads, so the list order can differ from arrival.
        _invocation("invocation_complete", T0 + 0.4, "encode"),
        _invocation("invocation_started", T0 + 0.1, "encode"),
        Event("model_load_started", T0 + 0.15, load),
        Event("model_load_complete", T0 + 0.35, load),
        _invocation("invocation_started", T0 + 0.5, "denoise"),
        *(
            _invocation("invocation_progress", T0 + 0.5 + 0.25 * k, "denoise", percentage=k / 4, has_image=True)
            for k in (1, 2, 4)
        ),
        _invocation("invocation_progress", T0 + 0.6, "denoise", percentage=None, has_image=False),
        _invocation("invocation_started", T0 + 0.2, "other", item_id=8),
        _invocation("invocation_complete", T0 + 1.6, "denoise"),
    ]
    item = {"started_at": "2026-10-17 12:00:00", "completed_at": "2026-10-17 12:00:02"}
    timing = build_timing(events, 7, T0, T0 + 1.7, item)
    assert timing.total_s == pytest.approx(1.7)
    assert [(n.node_id, n.type, round(n.t_start, 3), round(n.t_end, 3)) for n in timing.nodes] == [
        ("encode", "encode_type", 0.1, 0.4),
        ("denoise", "denoise_type", 0.5, 1.6),
    ]
    assert [(m.model, m.submodel, round(m.t_start, 3), round(m.t_end, 3)) for m in timing.model_loads] == [
        ("m", "transformer", 0.15, 0.35)
    ]
    assert timing.per_step_s == pytest.approx(0.25)
    assert (timing.queue_started_at, timing.queue_completed_at) == (item["started_at"], item["completed_at"])


def test_the_time_per_step_takes_the_count_of_the_node_that_sent_the_frames() -> None:
    # Two nodes state a count over their whole schedule. The denoise node sends most frames: six steps of 150 ms,
    # too fast for an inferred count to be trusted. A pass of twelve steps sent two; its larger count would halve the
    # time.
    events = [
        _invocation("invocation_started", T0, "denoise"),
        *(
            _invocation("invocation_progress", T0 + 0.15 * k, "denoise", percentage=k / 6, has_image=True)
            for k in range(1, 7)
        ),
        _invocation("invocation_complete", T0 + 1.0, "denoise"),
        _invocation("invocation_started", T0 + 1.0, "pass"),
        *(
            _invocation("invocation_progress", T0 + 1.0 + 0.5 * k, "pass", percentage=k / 12, has_image=True)
            for k in (6, 12)
        ),
        _invocation("invocation_complete", T0 + 7.0, "pass"),
    ]
    timing = build_timing(events, 7, T0, T0 + 7.1, {}, {"denoise": 6, "pass": 12})
    assert timing.per_step_s == pytest.approx(0.15)


def test_the_time_per_step_counts_the_steps_the_compiled_denoise_node_runs() -> None:
    # A case with `steps: 24` on Ideogram 4's V4_TURBO_12 preset runs twelve steps, here of 80 ms each. The 0.1 s
    # throttle let every other frame through: k/12 also reads as step 2k of 24, and inferred as step k/2 of 6.
    nodes = {
        "denoise_latents": {"id": "denoise_latents", "type": "ideogram4_denoise", "sampler_preset": "V4_TURBO_12"},
        "output": {"id": "output", "type": "ideogram4_l2i", "is_intermediate": False},
    }
    prepared = prepare_batch({"graph": {"id": "g", "nodes": nodes, "edges": []}, "data": []}, {}, {})
    events = [
        _invocation("invocation_started", T0, "denoise_latents"),
        *(
            _invocation("invocation_progress", T0 + 0.08 * k, "denoise_latents", percentage=k / 12, has_image=True)
            for k in range(2, 13, 2)
        ),
        _invocation("invocation_complete", T0 + 1.0, "denoise_latents"),
    ]
    timing = build_timing(events, 7, T0, T0 + 1.1, {}, prepared.step_counts)
    assert timing.per_step_s == pytest.approx(0.08)
