from types import SimpleNamespace

import pytest
import torch

from invokeai.backend.flux.denoise import denoise
from invokeai.backend.flux.schedulers import FLUX_SCHEDULER_MAP


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
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32)
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
