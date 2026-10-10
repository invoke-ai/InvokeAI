from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler

from invokeai.backend.flux.denoise import denoise
from invokeai.backend.flux.schedulers import _HAS_LCM, FLUX_SCHEDULER_MAP


class _FakeFluxModel:
    def __call__(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        y: torch.Tensor,
        timesteps: torch.Tensor,
        guidance: torch.Tensor,
        timestep_index: int,
        total_num_timesteps: int,
        controlnet_double_block_residuals: list[torch.Tensor] | None,
        controlnet_single_block_residuals: list[torch.Tensor] | None,
        ip_adapter_extensions: list[object],
        regional_prompting_extension: object,
    ) -> torch.Tensor:
        return torch.zeros_like(img)


class _ConstantFluxModel:
    def __init__(self, value: float) -> None:
        self.value = value
        self.timesteps: list[float] = []

    def __call__(self, img: torch.Tensor, timesteps: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        self.timesteps.append(float(timesteps[0]))
        return torch.full_like(img, self.value)


class _PromptFluxModel:
    def __init__(self) -> None:
        self.outputs: list[float] = []

    def __call__(self, img: torch.Tensor, txt: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        value = float(txt.mean())
        self.outputs.append(value)
        return torch.full_like(img, value)


class _SampleDependentFluxModel:
    def __init__(self) -> None:
        self.evaluations: list[tuple[float, torch.Tensor]] = []

    def __call__(self, img: torch.Tensor, timesteps: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        sigma = float(timesteps[0])
        self.evaluations.append((sigma, img.clone()))
        return 0.1 * img + 0.2 * sigma


class _RecordingInpaintExtension:
    def __init__(self) -> None:
        self.sigmas: list[float] = []

    def merge_intermediate_latents_with_init_latents(self, img: torch.Tensor, sigma: float) -> torch.Tensor:
        self.sigmas.append(sigma)
        return img


class _FakeDyPEExtension:
    def __init__(self) -> None:
        self.sigmas: list[float] = []

    def patch_model(self, model: object) -> tuple[object, None]:
        return object(), None

    def update_step_state(self, embedder: object, sigma: float) -> None:
        self.sigmas.append(sigma)


class _FakeScheduler:
    def __init__(self) -> None:
        self.config = SimpleNamespace(num_train_timesteps=1000)
        self.timesteps = torch.tensor([], dtype=torch.float32)
        self.sigmas = torch.tensor([], dtype=torch.float32)

    def set_timesteps(self, sigmas: list[float], device: torch.device) -> None:
        del device
        self.sigmas = torch.tensor([*sigmas, 0.0], dtype=torch.float32)
        self.timesteps = torch.tensor([900.0, 400.0], dtype=torch.float32)

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor) -> SimpleNamespace:
        del model_output, timestep
        return SimpleNamespace(prev_sample=sample)


class _FakeHeunScheduler:
    def __init__(self) -> None:
        self.config = SimpleNamespace(num_train_timesteps=1000)
        self.timesteps = torch.tensor([], dtype=torch.float32)
        self.sigmas = torch.tensor([], dtype=torch.float32)
        self.state_in_first_order = True
        self._step_index = 0

    def set_timesteps(self, sigmas: list[float], device: torch.device) -> None:
        del device
        # Duplicate each user-facing step to mimic a second-order scheduler.
        self.sigmas = torch.tensor([1.0, 1.0, 0.25, 0.25, 0.0], dtype=torch.float32)
        self.timesteps = torch.tensor([900.0, 850.0, 400.0, 350.0], dtype=torch.float32)
        self._step_index = 0
        self.state_in_first_order = True

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor) -> SimpleNamespace:
        del model_output, timestep
        self._step_index += 1
        self.state_in_first_order = self._step_index % 2 == 0
        return SimpleNamespace(prev_sample=sample)


class _FakePbar:
    def update(self, value: int) -> None:
        del value

    def close(self) -> None:
        return None


def _fake_tqdm(iterable=None, **kwargs):
    del kwargs
    if iterable is None:
        return _FakePbar()
    return iterable


def _build_regional_prompting_extension(batch_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        regional_text_conditioning=SimpleNamespace(
            t5_embeddings=torch.zeros(batch_size, 1, 4),
            t5_txt_ids=torch.zeros(batch_size, 1, 3),
            clip_embeddings=torch.zeros(batch_size, 4),
        )
    )


def test_denoise_euler_path_updates_dype_with_sigma(monkeypatch):
    monkeypatch.setattr("invokeai.backend.flux.denoise.tqdm", _fake_tqdm)

    model = _FakeFluxModel()
    dype_extension = _FakeDyPEExtension()
    img = torch.zeros(1, 2, 4)
    img_ids = torch.zeros(1, 2, 3)
    regional_prompting_extension = _build_regional_prompting_extension(batch_size=1)
    callback_steps: list[int] = []

    result = denoise(
        model=model,
        img=img,
        img_ids=img_ids,
        pos_regional_prompting_extension=regional_prompting_extension,
        neg_regional_prompting_extension=None,
        timesteps=[1.0, 0.5, 0.0],
        step_callback=lambda state: callback_steps.append(state.step),
        guidance=1.0,
        cfg_scale=[1.0, 1.0],
        inpaint_extension=None,
        controlnet_extensions=[],
        pos_ip_adapter_extensions=[],
        neg_ip_adapter_extensions=[],
        img_cond=None,
        img_cond_seq=None,
        img_cond_seq_ids=None,
        dype_extension=dype_extension,
        scheduler=None,
    )

    assert torch.equal(result, img)
    assert dype_extension.sigmas == [1.0, 0.5]
    assert callback_steps == [1, 2]


def test_denoise_scheduler_path_prefers_scheduler_sigmas_for_dype(monkeypatch):
    monkeypatch.setattr("invokeai.backend.flux.denoise.tqdm", _fake_tqdm)

    model = _FakeFluxModel()
    scheduler = _FakeScheduler()
    dype_extension = _FakeDyPEExtension()
    img = torch.zeros(1, 2, 4)
    img_ids = torch.zeros(1, 2, 3)
    regional_prompting_extension = _build_regional_prompting_extension(batch_size=1)

    denoise(
        model=model,
        img=img,
        img_ids=img_ids,
        pos_regional_prompting_extension=regional_prompting_extension,
        neg_regional_prompting_extension=None,
        timesteps=[1.0, 0.25, 0.0],
        step_callback=lambda state: None,
        guidance=1.0,
        cfg_scale=[1.0, 1.0],
        inpaint_extension=None,
        controlnet_extensions=[],
        pos_ip_adapter_extensions=[],
        neg_ip_adapter_extensions=[],
        img_cond=None,
        img_cond_seq=None,
        img_cond_seq_ids=None,
        dype_extension=dype_extension,
        scheduler=scheduler,
    )

    # Scheduler timesteps normalize to [0.9, 0.4], so this asserts the scheduler
    # sigma sequence is what DyPE actually consumes.
    assert dype_extension.sigmas == [1.0, 0.25]


def test_denoise_heun_scheduler_path_uses_internal_scheduler_sigmas(monkeypatch):
    monkeypatch.setattr("invokeai.backend.flux.denoise.tqdm", _fake_tqdm)

    model = _FakeFluxModel()
    scheduler = _FakeHeunScheduler()
    dype_extension = _FakeDyPEExtension()
    img = torch.zeros(1, 2, 4)
    img_ids = torch.zeros(1, 2, 3)
    regional_prompting_extension = _build_regional_prompting_extension(batch_size=1)
    callback_steps: list[int] = []

    denoise(
        model=model,
        img=img,
        img_ids=img_ids,
        pos_regional_prompting_extension=regional_prompting_extension,
        neg_regional_prompting_extension=None,
        timesteps=[1.0, 0.25, 0.0],
        step_callback=lambda state: callback_steps.append(state.step),
        guidance=1.0,
        cfg_scale=[1.0, 1.0],
        inpaint_extension=None,
        controlnet_extensions=[],
        pos_ip_adapter_extensions=[],
        neg_ip_adapter_extensions=[],
        img_cond=None,
        img_cond_seq=None,
        img_cond_seq_ids=None,
        dype_extension=dype_extension,
        scheduler=scheduler,
    )

    assert dype_extension.sigmas == [1.0, 1.0, 0.25, 0.25]
    assert callback_steps == [1, 2]


@pytest.mark.parametrize("scheduler_name", sorted(FLUX_SCHEDULER_MAP))
def test_denoise_real_flux_schedulers_update_dype_from_internal_sigma_schedule(monkeypatch, scheduler_name):
    monkeypatch.setattr("invokeai.backend.flux.denoise.tqdm", _fake_tqdm)

    model = _FakeFluxModel()
    scheduler = FLUX_SCHEDULER_MAP[scheduler_name](num_train_timesteps=1000)
    dype_extension = _FakeDyPEExtension()
    img = torch.zeros(1, 2, 4)
    img_ids = torch.zeros(1, 2, 3)
    regional_prompting_extension = _build_regional_prompting_extension(batch_size=1)
    callback_steps: list[int] = []

    denoise(
        model=model,
        img=img,
        img_ids=img_ids,
        pos_regional_prompting_extension=regional_prompting_extension,
        neg_regional_prompting_extension=None,
        timesteps=[1.0, 0.25, 0.0],
        step_callback=lambda state: callback_steps.append(state.step),
        guidance=1.0,
        cfg_scale=[1.0, 1.0],
        inpaint_extension=None,
        controlnet_extensions=[],
        pos_ip_adapter_extensions=[],
        neg_ip_adapter_extensions=[],
        img_cond=None,
        img_cond_seq=None,
        img_cond_seq_ids=None,
        dype_extension=dype_extension,
        scheduler=scheduler,
    )

    assert dype_extension.sigmas
    expected_sigmas = [float(sigma) for sigma in scheduler.sigmas[: len(dype_extension.sigmas)]]
    assert dype_extension.sigmas == expected_sigmas
    assert callback_steps


def _run_with_scheduler(scheduler, timesteps: list[float], **kwargs):
    model = kwargs.pop("model", _ConstantFluxModel(0.25))
    img = kwargs.pop("img", torch.tensor([[[1.0], [-0.5]]], dtype=torch.float32))
    callbacks = []
    result = denoise(
        model=model,
        img=img,
        img_ids=torch.zeros(1, img.shape[1], 3),
        pos_regional_prompting_extension=_build_regional_prompting_extension(batch_size=1),
        neg_regional_prompting_extension=None,
        timesteps=timesteps,
        step_callback=callbacks.append,
        guidance=1.0,
        cfg_scale=[1.0] * (len(timesteps) - 1),
        inpaint_extension=None,
        controlnet_extensions=[],
        pos_ip_adapter_extensions=[],
        neg_ip_adapter_extensions=[],
        img_cond=None,
        scheduler=scheduler,
        **kwargs,
    )
    return result, img, model, callbacks


@pytest.mark.parametrize("terminal_sigma", [0.0, 0.2])
def test_euler_uses_model_sigmas_and_requested_terminal_sigma(monkeypatch, terminal_sigma):
    monkeypatch.setattr("invokeai.backend.flux.denoise.tqdm", _fake_tqdm)
    schedule = [1.0, 0.7, 0.35, terminal_sigma]
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)

    result, img, model, callbacks = _run_with_scheduler(scheduler, schedule)

    expected = img + (terminal_sigma - schedule[0]) * 0.25
    assert torch.allclose(result, expected)
    assert len(model.timesteps) == len(schedule) - 1
    assert model.timesteps == pytest.approx(schedule[:-1])
    assert len(scheduler.timesteps) == len(schedule) - 1
    assert torch.equal(scheduler.sigmas[-1], torch.tensor(terminal_sigma))
    assert len(callbacks) == len(schedule) - 1
    assert scheduler.step_index == len(schedule) - 1


@pytest.mark.parametrize("terminal_sigma", [0.0, 0.2])
def test_default_scheduler_matches_explicit_euler_and_clean_preview(monkeypatch, terminal_sigma):
    monkeypatch.setattr("invokeai.backend.flux.denoise.tqdm", _fake_tqdm)
    schedule = [1.0, 0.7, 0.35, terminal_sigma]
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)

    scheduled, img, _, callbacks = _run_with_scheduler(scheduler, schedule)
    default, _, model, default_callbacks = _run_with_scheduler(None, schedule)

    assert torch.equal(scheduled, default)
    assert model.timesteps == pytest.approx(schedule[:-1])
    assert [callback.timestep for callback in default_callbacks] == [1000, 700, 350]
    assert [callback.step for callback in default_callbacks] == [1, 2, 3]
    assert all(callback.order == 1 and callback.total_steps == 3 for callback in default_callbacks)
    for explicit_callback, default_callback in zip(callbacks, default_callbacks, strict=True):
        assert torch.equal(explicit_callback.latents, default_callback.latents)
    expected_clean = img - 0.25
    assert all(torch.allclose(callback.latents, expected_clean) for callback in callbacks)


def test_heun_uses_custom_schedule_and_completes_each_user_step(monkeypatch):
    monkeypatch.setattr("invokeai.backend.flux.denoise.tqdm", _fake_tqdm)
    schedule = [1.0, 0.7, 0.35, 0.2]
    scheduler = FlowMatchHeunDiscreteScheduler(num_train_timesteps=1000)

    result, img, model, callbacks = _run_with_scheduler(scheduler, schedule)

    expected = img + (schedule[-1] - schedule[0]) * 0.25
    expected_timesteps = torch.tensor([1000.0, 700.0, 700.0, 350.0, 350.0])
    expected_sigmas = torch.tensor([1.0, 0.7, 0.7, 0.35, 0.35, 0.2])
    assert torch.allclose(result, expected, atol=1e-6, rtol=0)
    assert torch.equal(scheduler.timesteps.cpu(), expected_timesteps)
    assert torch.equal(scheduler.sigmas.cpu(), expected_sigmas)
    assert len(model.timesteps) == 2 * (len(schedule) - 1) - 1
    assert [callback.step for callback in callbacks] == [1, 2, 3]
    assert [callback.order for callback in callbacks] == [2, 2, 1]


def test_heun_partial_start_and_end_use_custom_sigmas(monkeypatch):
    monkeypatch.setattr("invokeai.backend.flux.denoise.tqdm", _fake_tqdm)
    schedule = [0.7, 0.45, 0.2]
    scheduler = FLUX_SCHEDULER_MAP["heun"](num_train_timesteps=1000)

    result, img, model, callbacks = _run_with_scheduler(scheduler, schedule)

    assert torch.allclose(result, img + (0.2 - 0.7) * 0.25, atol=1e-6, rtol=0)
    assert torch.equal(scheduler.timesteps.cpu(), torch.tensor([700.0, 450.0, 450.0]))
    assert torch.equal(scheduler.sigmas.cpu(), torch.tensor([0.7, 0.45, 0.45, 0.2]))
    assert len(model.timesteps) == 3
    assert [callback.step for callback in callbacks] == [1, 2]
    assert [callback.order for callback in callbacks] == [2, 1]


def test_heun_preserves_cfg_and_inpainting_callback_behavior(monkeypatch):
    monkeypatch.setattr("invokeai.backend.flux.denoise.tqdm", _fake_tqdm)
    img = torch.tensor([[[1.0]]])
    schedule = [1.0, 0.7, 0.35, 0.0]
    scheduler = FlowMatchHeunDiscreteScheduler(num_train_timesteps=1000)
    model = _PromptFluxModel()
    extension = _RecordingInpaintExtension()
    prompt = _build_regional_prompting_extension(batch_size=1)
    negative_prompt = _build_regional_prompting_extension(batch_size=1)
    prompt.regional_text_conditioning.t5_embeddings.fill_(1.0)
    negative_prompt.regional_text_conditioning.t5_embeddings.fill_(-1.0)
    callbacks = []

    result = denoise(
        model=model,
        img=img,
        img_ids=torch.zeros(1, 1, 3),
        pos_regional_prompting_extension=prompt,
        neg_regional_prompting_extension=negative_prompt,
        timesteps=schedule,
        step_callback=callbacks.append,
        guidance=1.0,
        cfg_scale=[2.0] * (len(schedule) - 1),
        inpaint_extension=extension,
        controlnet_extensions=[],
        pos_ip_adapter_extensions=[],
        neg_ip_adapter_extensions=[],
        img_cond=None,
        scheduler=scheduler,
    )

    guided_prediction = -1.0 + 2.0 * (1.0 - -1.0)
    assert torch.allclose(result, img + (schedule[-1] - schedule[0]) * guided_prediction)
    assert model.outputs == [1.0, -1.0] * 5
    assert [callback.step for callback in callbacks] == [1, 2, 3]
    assert [callback.order for callback in callbacks] == [2, 2, 1]
    assert all(torch.allclose(callback.latents, img - guided_prediction) for callback in callbacks)
    assert extension.sigmas == pytest.approx([0.7, 0.7, 0.0, 0.35, 0.35, 0.0, 0.0, 0.0])


def test_heun_keeps_second_order_correction_for_sample_dependent_predictions(monkeypatch):
    monkeypatch.setattr("invokeai.backend.flux.denoise.tqdm", _fake_tqdm)
    schedule = [0.9, 0.6, 0.3, 0.1]
    scheduler = FLUX_SCHEDULER_MAP["heun"](num_train_timesteps=1000)
    model = _SampleDependentFluxModel()

    result, img, _, callbacks = _run_with_scheduler(scheduler, schedule, model=model)

    expected = img.clone()
    for index, sigma in enumerate(schedule[:-2]):
        sigma_next = schedule[index + 1]
        first_prediction = 0.1 * expected + 0.2 * sigma
        euler_sample = expected + (sigma_next - sigma) * first_prediction
        second_prediction = 0.1 * euler_sample + 0.2 * sigma_next
        expected = expected + (sigma_next - sigma) * (first_prediction + second_prediction) / 2
    terminal_prediction = 0.1 * expected + 0.2 * schedule[-2]
    expected = expected + (schedule[-1] - schedule[-2]) * terminal_prediction

    assert torch.allclose(result, expected, atol=1e-6, rtol=0)
    assert len(model.evaluations) == 2 * (len(schedule) - 1) - 1
    assert [callback.order for callback in callbacks] == [2, 2, 1]


def test_scheduler_preview_preserves_cfg_and_inpainting_callback(monkeypatch):
    monkeypatch.setattr("invokeai.backend.flux.denoise.tqdm", _fake_tqdm)
    img = torch.tensor([[[1.0]]])
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    model = _PromptFluxModel()
    extension = _RecordingInpaintExtension()
    prompt = _build_regional_prompting_extension(batch_size=1)
    negative_prompt = _build_regional_prompting_extension(batch_size=1)
    prompt.regional_text_conditioning.t5_embeddings.fill_(1.0)
    negative_prompt.regional_text_conditioning.t5_embeddings.fill_(-1.0)
    callbacks = []
    result = denoise(
        model=model,
        img=img,
        img_ids=torch.zeros(1, 1, 3),
        pos_regional_prompting_extension=prompt,
        neg_regional_prompting_extension=negative_prompt,
        timesteps=[1.0, 0.5, 0.0],
        step_callback=callbacks.append,
        guidance=1.0,
        cfg_scale=[2.0, 2.0],
        inpaint_extension=extension,
        controlnet_extensions=[],
        pos_ip_adapter_extensions=[],
        neg_ip_adapter_extensions=[],
        img_cond=None,
        scheduler=scheduler,
    )

    guided_prediction = -1.0 + 2.0 * (1.0 - -1.0)
    assert torch.allclose(result, img + (0.0 - 1.0) * guided_prediction)
    assert model.outputs == [1.0, -1.0, 1.0, -1.0]
    assert [callback.step for callback in callbacks] == [1, 2]
    assert all(torch.allclose(callback.latents, img - guided_prediction) for callback in callbacks)
    assert extension.sigmas == [0.5, 0.0, 0.0, 0.0]


@pytest.mark.skipif(not _HAS_LCM, reason="FlowMatchLCMScheduler not available in this Diffusers version")
def test_lcm_preview_uses_the_model_evaluation_sample_and_sigma(monkeypatch):
    monkeypatch.setattr("invokeai.backend.flux.denoise.tqdm", _fake_tqdm)
    scheduler = FLUX_SCHEDULER_MAP["lcm"](num_train_timesteps=1000)
    result, img, model, callbacks = _run_with_scheduler(scheduler, [1.0, 0.5, 0.0])

    current_sigma = scheduler.timesteps[0] / scheduler.config.num_train_timesteps
    expected_preview = img - current_sigma * 0.25
    assert result.shape == img.shape
    assert len(model.timesteps) == len(scheduler.timesteps) == len(callbacks) == 2
    assert model.timesteps == pytest.approx(
        [float(timestep / scheduler.config.num_train_timesteps) for timestep in scheduler.timesteps]
    )
    assert torch.allclose(callbacks[0].latents, expected_preview)


@pytest.mark.parametrize("schedule", [[], [0.4]])
def test_default_scheduler_without_intervals_returns_input(schedule):
    result, img, model, callbacks = _run_with_scheduler(None, schedule)

    assert result is img
    assert model.timesteps == []
    assert callbacks == []
