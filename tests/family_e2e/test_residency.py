from scripts.family_e2e.records import Residency
from scripts.family_e2e.residency import ResidencyTracker, parse_log, resident, weightless

PREFIX = "[2026-10-17 12:00:00,000]::[ModelManagerService]::INFO --> [MODEL CACHE] "
MODEL = "6dc7b49c-2a39-4901-a38b-4b291320b42f"
VAE = "4d85e528-e888-4e72-9539-41612778409b"


def loaded(key: str, fraction: float, device: str = "cuda device #0", total_mb: float = 7600.0) -> str:
    # The format of ModelCache._load_locked_model, including its percent formatting.
    return (
        f"{PREFIX}Loaded model '{key}' (Krea2Transformer2DModel) onto {device} in 3.21s. "
        f"Total model size: {total_mb:.2f}MB, VRAM: {total_mb * fraction:.2f}MB ({fraction:.1%})\n"
    )


def short(key: str) -> str:
    return (
        "[2026-10-17 12:00:00,000]::[ModelManagerService]::WARNING --> [MODEL CACHE] "
        f"VRAM budget for '{key}' is short by 812MB even after offloading (working memory reservation: 3072MB); "
        "the model will run with minimum weights resident. Other models still in VRAM: none.\n"
    )


def test_load_lines_parse_into_model_submodel_and_percent() -> None:
    text = (
        loaded(f"{MODEL}:transformer", 1.0)
        + f"{PREFIX}Loaded model '{VAE}' (AutoencoderKL) onto cpu device (cpu_only); skipping VRAM load\n"
        + loaded(VAE, 0.5, device="cuda device", total_mb=160.0)
        + short(f"{MODEL}:transformer")
    )
    window = parse_log(text)
    assert [(r.model, r.submodel, r.vram_mb, r.total_mb, r.percent) for r in window.loads] == [
        (MODEL, "transformer", 7600.0, 7600.0, 100.0),
        (VAE, None, 80.0, 160.0, 50.0),
    ]
    assert window.short == [f"{MODEL}:transformer"]


def test_warm_runs_without_load_lines_stay_valid_after_a_full_load() -> None:
    tracker = ResidencyTracker()
    residency, reasons = tracker.observe(parse_log(loaded(f"{MODEL}:transformer", 1.0)), [MODEL])
    assert reasons == []
    residency, reasons = tracker.observe(parse_log("nothing about models\n"), [MODEL])
    assert reasons == []
    assert [(r.model, r.percent) for r in residency] == [(MODEL, 100.0)]


def test_partial_load_invalidates_this_run_and_later_runs_without_a_new_line() -> None:
    tracker = ResidencyTracker()
    _, reasons = tracker.observe(parse_log(loaded(f"{MODEL}:transformer", 0.874)), [MODEL])
    assert reasons == [f"residency: {MODEL}:transformer loaded with 87.4% of its weights in VRAM"]
    _, reasons = tracker.observe(parse_log(""), [MODEL])
    assert reasons == [f"residency: {MODEL}:transformer at 87.4% since an earlier run"]
    _, reasons = tracker.observe(parse_log(loaded(f"{MODEL}:transformer", 1.0)), [MODEL])
    assert reasons == []


def test_rounding_to_one_decimal_counts_as_full_only_at_100_0() -> None:
    tracker = ResidencyTracker()
    assert tracker.observe(parse_log(loaded(MODEL, 0.99951)), [MODEL])[1] == []
    assert tracker.observe(parse_log(loaded(MODEL, 0.9994)), [MODEL])[1] != []


def test_a_weightless_tokenizer_load_is_resident_but_a_partial_weighted_load_is_not() -> None:
    # The lines of a FLUX.1 dev (int8) run: its T5 encoder's tokenizer has no weights, the cache logs it at 0.0 %.
    tokenizer = (
        f"{PREFIX}Loaded model '{MODEL}:tokenizer_2' (T5Tokenizer) onto cuda device #0 in 0.00s. "
        "Total model size: 0.03MB, VRAM: 0.00MB (0.0%)\n"
    )
    tracker = ResidencyTracker()

    residency, reasons = tracker.observe(parse_log(tokenizer + loaded(f"{MODEL}:text_encoder_2", 1.0)), [MODEL])
    assert reasons == []
    # The record keeps what the cache logged.
    assert [(r.submodel, r.total_mb, r.percent) for r in residency] == [
        ("tokenizer_2", 0.03, 0.0),
        ("text_encoder_2", 7600.0, 100.0),
    ]

    _, reasons = tracker.observe(parse_log(tokenizer + loaded(VAE, 0.5, total_mb=160.0)), [MODEL])
    assert reasons == [f"residency: {VAE} loaded with 50.0% of its weights in VRAM"]


def test_a_record_written_before_sizes_were_kept_is_never_weightless() -> None:
    # Without a logged size a 0.0 % load cannot be told from weights left entirely in RAM.
    old = Residency(model=MODEL, submodel="transformer", vram_mb=0.0, total_mb=None, percent=0.0)
    assert not weightless(old)
    assert not resident(old)


def test_a_weightless_load_alone_leaves_residency_unknown() -> None:
    tokenizer = (
        f"{PREFIX}Loaded model '{MODEL}:tokenizer_2' (T5Tokenizer) onto cuda device #0 in 0.00s. "
        "Total model size: 0.03MB, VRAM: 0.00MB (0.0%)\n"
    )
    _, reasons = ResidencyTracker().observe(parse_log(tokenizer), [MODEL])
    assert reasons == ["residency unknown: no load with weights for the graph's models since the last cache flush"]


def test_short_budget_warning_invalidates_even_with_full_lines() -> None:
    tracker = ResidencyTracker()
    text = short(f"{MODEL}:transformer") + loaded(f"{MODEL}:transformer", 1.0)
    _, reasons = tracker.observe(parse_log(text), [MODEL])
    assert reasons == [f"residency: VRAM budget short for {MODEL}:transformer"]


def test_without_any_line_for_the_graphs_models_residency_is_unknown() -> None:
    tracker = ResidencyTracker()
    unknown = "residency unknown: no load with weights for the graph's models since the last cache flush"
    assert tracker.observe(parse_log(""), [MODEL])[1] == [unknown]
    # Another model's full load says nothing about this graph's models.
    assert tracker.observe(parse_log(loaded(VAE, 1.0)), [MODEL])[1] == [unknown]
    tracker.observe(parse_log(loaded(f"{MODEL}:transformer", 1.0)), [MODEL])
    tracker.reset()
    assert tracker.observe(parse_log(""), [MODEL])[1] == [unknown]


def test_partial_load_of_a_model_outside_the_graph_still_invalidates_the_run() -> None:
    tracker = ResidencyTracker()
    text = loaded(f"{MODEL}:transformer", 1.0) + loaded(VAE, 0.3)
    residency, reasons = tracker.observe(parse_log(text), [MODEL])
    assert [r.model for r in residency] == [MODEL]
    assert reasons == [f"residency: {VAE} loaded with 30.0% of its weights in VRAM"]
