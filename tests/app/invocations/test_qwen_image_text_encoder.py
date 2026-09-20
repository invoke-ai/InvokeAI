"""Tests for the Qwen Image text encoder prompt building and image resizing."""

import gc
import json
import threading
import traceback
import weakref
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from invokeai.app.invocations.model import ModelIdentifierField, QwenVLEncoderField
from invokeai.app.invocations.qwen_image_text_encoder import (
    _GENERATE_DROP_IDX,
    QwenImageTextEncoderInvocation,
    _build_prompt,
    _read_checkpoint,
)
from invokeai.backend.model_manager.load.model_cache.model_cache import MB, MODEL_LOAD_LOCK
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.backend.util.devices import TorchDevice


class TestBuildPrompt:
    """Test the _build_prompt function for edit vs generate modes."""

    def test_no_images_uses_generate_template(self):
        """With 0 images, should use the generate (txt2img) template with no vision placeholder."""
        prompt = _build_prompt("a beautiful sunset", 0)
        assert "a beautiful sunset" in prompt
        assert "<|im_start|>assistant" in prompt
        # Generate mode: no vision placeholders, uses the "describe the image" system prompt
        assert "<|vision_start|>" not in prompt
        assert "Describe the image by detailing" in prompt

    def test_no_images_does_not_use_edit_template(self):
        """With 0 images, should NOT use the edit system prompt."""
        prompt = _build_prompt("a beautiful sunset", 0)
        assert "Describe the key features of the input image" not in prompt

    def test_edit_mode_one_image(self):
        """With 1 image, should use the edit template with one vision placeholder."""
        prompt = _build_prompt("change hair to red", 1)
        assert "Describe the key features of the input image" in prompt
        assert prompt.count("<|vision_start|><|image_pad|><|vision_end|>") == 1
        assert "change hair to red" in prompt
        # Should NOT use the generate system prompt
        assert "Describe the image by detailing" not in prompt

    def test_edit_mode_multiple_images(self):
        """With multiple images, should include one placeholder per image."""
        prompt = _build_prompt("combine these images", 3)
        assert prompt.count("<|vision_start|><|image_pad|><|vision_end|>") == 3
        assert "combine these images" in prompt

    def test_generate_template_has_correct_structure(self):
        """Generate template should have system + user + assistant roles."""
        prompt = _build_prompt("test prompt", 0)
        assert prompt.startswith("<|im_start|>system\n")
        assert "<|im_end|>\n<|im_start|>user\n" in prompt
        assert prompt.endswith("<|im_start|>assistant\n")

    def test_edit_template_has_correct_structure(self):
        """Edit template should have system + user (with image) + assistant roles."""
        prompt = _build_prompt("test prompt", 1)
        assert prompt.startswith("<|im_start|>system\n")
        assert "<|im_end|>\n<|im_start|>user\n" in prompt
        assert "<|vision_start|>" in prompt
        assert prompt.endswith("<|im_start|>assistant\n")

    def test_prompt_special_characters(self):
        """Prompt with special characters should be included verbatim."""
        prompt = _build_prompt("add {curly} braces & <angle> brackets", 0)
        assert "add {curly} braces & <angle> brackets" in prompt


class TestResizeForVLEncoder:
    """Test the image resizing logic for the VL encoder."""

    def test_large_image_is_resized(self):
        """A large image should be resized to ~target_pixels."""
        img = Image.new("RGB", (2048, 2048))
        resized = QwenImageTextEncoderInvocation._resize_for_vl_encoder(img, target_pixels=512 * 512)
        w, h = resized.size
        # Should be much smaller than original
        assert w < 2048
        assert h < 2048
        # Total pixels should be approximately target
        assert abs(w * h - 512 * 512) < 10000  # within ~10k pixels

    def test_small_image_is_resized(self):
        """A small image should also be resized to ~target_pixels."""
        img = Image.new("RGB", (64, 64))
        resized = QwenImageTextEncoderInvocation._resize_for_vl_encoder(img, target_pixels=512 * 512)
        w, h = resized.size
        # Should be larger than original
        assert w > 64
        assert h > 64

    def test_aspect_ratio_preserved(self):
        """Aspect ratio should be approximately preserved."""
        img = Image.new("RGB", (800, 400))  # 2:1 aspect ratio
        resized = QwenImageTextEncoderInvocation._resize_for_vl_encoder(img, target_pixels=512 * 512)
        w, h = resized.size
        original_ratio = 800 / 400  # 2.0
        new_ratio = w / h
        # Allow some deviation due to rounding to multiples of 32
        assert abs(new_ratio - original_ratio) < 0.3

    def test_dimensions_are_multiples_of_32(self):
        """Output dimensions should be multiples of 32."""
        img = Image.new("RGB", (1000, 750))
        resized = QwenImageTextEncoderInvocation._resize_for_vl_encoder(img, target_pixels=512 * 512)
        w, h = resized.size
        assert w % 32 == 0
        assert h % 32 == 0

    def test_square_image(self):
        """A square image should produce approximately square output."""
        img = Image.new("RGB", (1024, 1024))
        resized = QwenImageTextEncoderInvocation._resize_for_vl_encoder(img, target_pixels=512 * 512)
        w, h = resized.size
        assert abs(w - h) <= 32  # within one grid step

    def test_portrait_image(self):
        """A portrait image should produce portrait output."""
        img = Image.new("RGB", (600, 1200))
        resized = QwenImageTextEncoderInvocation._resize_for_vl_encoder(img, target_pixels=512 * 512)
        w, h = resized.size
        assert h > w  # should remain portrait

    def test_landscape_image(self):
        """A landscape image should produce landscape output."""
        img = Image.new("RGB", (1200, 600))
        resized = QwenImageTextEncoderInvocation._resize_for_vl_encoder(img, target_pixels=512 * 512)
        w, h = resized.size
        assert w > h  # should remain landscape


def _read_lock_held() -> bool:
    return MODEL_LOAD_LOCK._readers > 0 and not MODEL_LOAD_LOCK._writer_active


def _write_lock_held() -> bool:
    return MODEL_LOAD_LOCK._writer_active


def _tiny_config() -> Qwen2_5_VLConfig:
    """A Qwen2.5-VL config small enough to build and load on the CPU in well under a second."""
    return Qwen2_5_VLConfig(
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 512,
            "max_position_embeddings": 256,
            "tie_word_embeddings": False,
            "rope_scaling": {"type": "mrope", "mrope_section": [2, 3, 3]},
        },
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 64,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "in_channels": 3,
            "window_size": 112,
            "fullatt_block_indexes": [1],
        },
    )


def _write_sharded_checkpoint(text_encoder_dir: Path, total_size: int) -> dict[str, torch.Tensor]:
    """A two-shard safetensors checkpoint small enough to be read for real, with a config so the loader can build
    the model config from it. `total_size` is what `calc_model_size_by_fs` reports for it."""
    text_encoder_dir.mkdir(parents=True)
    _tiny_config().save_pretrained(text_encoder_dir)
    tensors = {
        "a.weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
        "b.bias": torch.ones(4, dtype=torch.bfloat16),
    }
    save_file({"a.weight": tensors["a.weight"]}, text_encoder_dir / "model-00001-of-00002.safetensors")
    save_file({"b.bias": tensors["b.bias"]}, text_encoder_dir / "model-00002-of-00002.safetensors")
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": {"a.weight": "model-00001-of-00002.safetensors", "b.bias": "model-00002-of-00002.safetensors"},
    }
    (text_encoder_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    return tensors


class TestReadCheckpoint:
    def test_reads_every_shard_named_by_the_index(self, tmp_path: Path):
        tensors = _write_sharded_checkpoint(tmp_path / "text_encoder", total_size=1)

        state_dict = _read_checkpoint(tmp_path / "text_encoder")

        assert state_dict is not None and set(state_dict) == set(tensors)
        assert all(torch.equal(state_dict[k], tensors[k]) for k in tensors)

    def test_reads_an_unsharded_checkpoint(self, tmp_path: Path):
        (tmp_path / "text_encoder").mkdir()
        save_file({"w": torch.zeros(3)}, tmp_path / "text_encoder" / "model.safetensors")

        state_dict = _read_checkpoint(tmp_path / "text_encoder")

        assert state_dict is not None and list(state_dict) == ["w"]

    def test_returns_none_without_safetensors_shards(self, tmp_path: Path):
        """A `.bin` checkpoint is left to transformers to read (under the lock, as before)."""
        (tmp_path / "text_encoder").mkdir()
        (tmp_path / "text_encoder" / "pytorch_model.bin").write_bytes(b"")

        assert _read_checkpoint(tmp_path / "text_encoder") is None


class TestQuantizedEncoderLoad:
    """The BitsAndBytes path bypasses the model cache, so it must ask the cache for VRAM itself and load onto the
    worker's execution device explicitly (issue #9147: `device_map="auto"` spilled to the CPU because the cached
    transformer and VAE still filled the card, and BnB int8 refused to run that way).
    """

    TOTAL_SIZE = 16 * 2**30  # bf16 Qwen2.5-VL-7B on disk

    @staticmethod
    def _make_invocation(quantization: str) -> QwenImageTextEncoderInvocation:
        encoder = ModelIdentifierField(
            key="enc", hash="h", name="qwen-vl", base=BaseModelType.QwenImage, type=ModelType.QwenVLEncoder
        )
        return QwenImageTextEncoderInvocation(
            prompt="a cat",
            qwen_vl_encoder=QwenVLEncoderField(tokenizer=encoder, text_encoder=encoder),
            quantization=quantization,
        )

    def _make_context(
        self, tmp_path: Path, events: list[str], single_file: bool = False, vram_available: int | None = None
    ) -> tuple[MagicMock, dict[str, torch.Tensor]]:
        """`vram_available` is what `make_room_in_vram` reports after offloading; by default the request is met."""
        tensors: dict[str, torch.Tensor] = {}
        if single_file:
            model_root = tmp_path / "encoder.safetensors"
            model_root.write_bytes(b"")
        else:
            model_root = tmp_path / "qwen-vl"
            tensors = _write_sharded_checkpoint(model_root / "text_encoder", total_size=self.TOTAL_SIZE)

        context = MagicMock()
        context.models.get_absolute_path.return_value = model_root

        def make_room(vram_bytes_needed: int) -> int:
            # The offload is a VRAM move like any other, so it runs under the model-load *read* lock.
            events.append("make_room" if _read_lock_held() else "make_room(wrong lock)")
            return vram_bytes_needed if vram_available is None else vram_available

        context.models.make_room_in_vram.side_effect = make_room
        return context, tensors

    @pytest.mark.parametrize(("quantization", "ratio"), [("int8", 0.6), ("nf4", 0.4)])
    def test_makes_room_in_vram_before_loading_onto_the_execution_device(
        self, tmp_path: Path, quantization: str, ratio: float
    ):
        events: list[str] = []
        context, tensors = self._make_context(tmp_path, events)
        fake_model = MagicMock()
        device = torch.device("cuda:1")
        seen: dict = {}

        def fake_from_pretrained(path, **kwargs):
            events.append("from_pretrained")
            seen.update(kwargs)
            seen["path"] = path
            # The construction must run under the model-load *write* lock: it installs process-global patches no
            # other construction may overlap, and a concurrent cache construction would hijack its parameter
            # assignment onto the meta device (see TestTransformersLoadPathAssumptions).
            seen["write_locked"] = _write_lock_held()
            return fake_model

        with (
            patch.object(Qwen2_5_VLForConditionalGeneration, "from_pretrained", side_effect=fake_from_pretrained),
            patch.object(TorchDevice, "choose_torch_device", return_value=device),
        ):
            text_encoder, returned_device, cleanup = self._make_invocation(quantization)._load_quantized_encoder(
                context
            )

        assert events == ["make_room", "from_pretrained"]
        context.models.make_room_in_vram.assert_called_once_with(int(self.TOTAL_SIZE * ratio))
        assert seen["device_map"] == {"": device}, "must never use device_map='auto'"
        assert seen["write_locked"]
        # The checkpoint is read by the invocation (outside the lock) and handed over, not read by transformers.
        assert seen["path"] is None
        assert set(seen["state_dict"]) == set(tensors)
        assert all(torch.equal(seen["state_dict"][k], tensors[k]) for k in tensors)
        assert isinstance(seen["config"], Qwen2_5_VLConfig)
        assert seen["quantization_config"].load_in_8bit == (quantization == "int8")
        assert seen["quantization_config"].load_in_4bit == (quantization == "nf4")
        assert text_encoder is fake_model
        assert returned_device == device
        context.logger.warning.assert_not_called()
        cleanup()

    def test_checkpoint_is_read_outside_the_model_load_lock(self, tmp_path: Path):
        """The disk read is the slow part of the load on anything but NVMe, and while MODEL_LOAD_LOCK is held (or
        wanted by a writer) other workers' VRAM moves and cold constructions wait. Only the offload and the
        construction may run under it; a construction elsewhere must be able to take the write lock while the
        checkpoint is read. The read also comes *after* the offload, so the offload's RAM cannot evict its pages."""
        events: list[str] = []
        context, _ = self._make_context(tmp_path, events)
        writer_acquired = threading.Event()
        acquired_during_read: list[bool] = []

        def fake_read(encoder_path: Path):
            events.append("read" if not (_read_lock_held() or _write_lock_held()) else "read(locked)")

            def construction_on_another_worker():
                with MODEL_LOAD_LOCK.write_lock():
                    writer_acquired.set()

            threading.Thread(target=construction_on_another_worker).start()
            acquired_during_read.append(writer_acquired.wait(timeout=5))
            return {}

        def fake_from_pretrained(path, **kwargs):
            events.append("from_pretrained" if _write_lock_held() else "from_pretrained(wrong lock)")
            return MagicMock()

        with (
            patch("invokeai.app.invocations.qwen_image_text_encoder._read_checkpoint", side_effect=fake_read),
            patch.object(Qwen2_5_VLForConditionalGeneration, "from_pretrained", side_effect=fake_from_pretrained),
            patch.object(TorchDevice, "choose_torch_device", return_value=torch.device("cuda")),
        ):
            self._make_invocation("int8")._load_quantized_encoder(context)

        assert acquired_during_read == [True], "a construction was blocked for the duration of the checkpoint read"
        assert events == ["make_room", "read", "from_pretrained"]

    def test_warns_when_less_than_the_estimate_could_be_made_available(self, tmp_path: Path):
        """`make_room_in_vram` cannot offload locked models, so it reports what is actually available after
        offloading; a shortfall must be visible in the log rather than surfacing only as a later OOM."""
        events: list[str] = []
        context, _ = self._make_context(tmp_path, events, vram_available=1000 * MB)

        with (
            patch.object(Qwen2_5_VLForConditionalGeneration, "from_pretrained", return_value=MagicMock()),
            patch.object(TorchDevice, "choose_torch_device", return_value=torch.device("cuda:0")),
        ):
            self._make_invocation("int8")._load_quantized_encoder(context)

        context.logger.warning.assert_called_once()
        message = context.logger.warning.call_args.args[0]
        assert "1000 MB" in message
        assert f"{int(self.TOTAL_SIZE * 0.6) / MB:.0f} MB" in message
        assert "cuda:0" in message

    @pytest.mark.parametrize(
        ("error_type", "error_message"), [(torch.OutOfMemoryError, "CUDA out of memory"), (ValueError, "bad shard")]
    )
    def test_failed_load_releases_the_partial_model_before_the_error_escapes(
        self, tmp_path: Path, error_type: type[Exception], error_message: str
    ):
        """When `from_pretrained` raises, the partially built model is referenced only by the traceback's frames.
        Left alone, its weights stay allocated until the session processor drops the exception and *reserved* by
        torch after that, so the next load under-budgets - the same leak `_encode` guards against on the success
        path. An OOM additionally comes back with the numbers the user needs to act on."""
        events: list[str] = []
        context, _ = self._make_context(tmp_path, events, vram_available=4000 * MB)
        partial_ref: list[weakref.ref] = []
        alive_at_empty_cache: list[bool] = []

        def fake_from_pretrained(path, **kwargs):
            partial_model = torch.nn.Linear(2, 2)  # referenced only by this frame once we raise
            partial_model.cycle = partial_model  # real models carry cycles (hooks, recorders): refcounting won't do
            partial_ref.append(weakref.ref(partial_model))
            raise error_type(error_message)

        def empty_cache():
            alive_at_empty_cache.append(partial_ref[0]() is not None)

        with (
            patch.object(Qwen2_5_VLForConditionalGeneration, "from_pretrained", side_effect=fake_from_pretrained),
            patch.object(TorchDevice, "choose_torch_device", return_value=torch.device("cuda:0")),
            patch.object(TorchDevice, "empty_cache", side_effect=empty_cache),
            pytest.raises(error_type) as excinfo,
        ):
            self._make_invocation("int8")._load_quantized_encoder(context)

        assert alive_at_empty_cache == [False], "empty_cache() ran while the traceback still referenced the model"
        assert partial_ref[0]() is None
        # The loader's own frame outlives the failure (it is in the traceback), so it must have dropped the mapped
        # checkpoint itself - `clear_frames` cannot clear an executing frame.
        loader_locals = next(
            f.f_locals for f, _ in traceback.walk_tb(excinfo.tb) if f.f_code.co_name == "_load_quantized_encoder"
        )
        assert loader_locals["state_dict"] is None
        if error_type is torch.OutOfMemoryError:
            assert isinstance(excinfo.value.__cause__, torch.OutOfMemoryError)
            assert error_message in str(excinfo.value.__cause__)
            message = str(excinfo.value)
            assert "cuda:0" in message
            assert f"{int(self.TOTAL_SIZE * 0.6) / MB:.0f} MB" in message
            assert "4000 MB" in message
            assert "nf4" in message
        else:
            assert str(excinfo.value) == error_message, "only an OOM is re-described"

    def test_cpu_execution_device_makes_no_room_and_logs_no_shortfall(self, tmp_path: Path):
        """A CPU execution device has no VRAM to make room in (the cache's `make_room_in_vram` reports 0 there), and
        recent bitsandbytes can quantize on the CPU, so the load proceeds without a spurious shortfall warning."""
        events: list[str] = []
        context, _ = self._make_context(tmp_path, events)
        seen: dict = {}

        def fake_from_pretrained(path, **kwargs):
            seen.update(kwargs)
            return MagicMock()

        with (
            patch.object(Qwen2_5_VLForConditionalGeneration, "from_pretrained", side_effect=fake_from_pretrained),
            patch.object(TorchDevice, "choose_torch_device", return_value=torch.device("cpu")),
        ):
            self._make_invocation("int8")._load_quantized_encoder(context)

        context.models.make_room_in_vram.assert_not_called()
        context.logger.warning.assert_not_called()
        assert seen["device_map"] == {"": torch.device("cpu")}

    def test_unsizeable_checkpoint_warns_instead_of_silently_requesting_zero_bytes(self, tmp_path: Path):
        """`calc_model_size_by_fs` reports 0 for weights it cannot size, and a request for 0 bytes is a silent no-op
        that brings issue #9147 straight back. Such a checkpoint has no safetensors shards either, so transformers
        reads it itself."""
        model_root = tmp_path / "qwen-vl"
        text_encoder_dir = model_root / "text_encoder"
        text_encoder_dir.mkdir(parents=True)
        _tiny_config().save_pretrained(text_encoder_dir)
        (text_encoder_dir / "pytorch_model.pth").write_bytes(b"\0" * 16)
        context = MagicMock()
        context.models.get_absolute_path.return_value = model_root
        seen: dict = {}

        def fake_from_pretrained(path, **kwargs):
            seen.update(kwargs)
            seen["path"] = path
            return MagicMock()

        with (
            patch.object(Qwen2_5_VLForConditionalGeneration, "from_pretrained", side_effect=fake_from_pretrained),
            patch.object(TorchDevice, "choose_torch_device", return_value=torch.device("cuda:0")),
        ):
            self._make_invocation("int8")._load_quantized_encoder(context)

        context.models.make_room_in_vram.assert_not_called()
        context.logger.warning.assert_called_once()
        assert str(text_encoder_dir) in context.logger.warning.call_args.args[0]
        assert seen["path"] == str(text_encoder_dir)
        assert seen["state_dict"] is None

    def test_single_file_checkpoint_falls_back_to_the_cache_without_making_room(self, tmp_path: Path):
        """A single-file encoder cannot be BnB-quantized; it goes through the cache, which makes its own room."""
        events: list[str] = []
        context, _ = self._make_context(tmp_path, events, single_file=True)
        invocation = self._make_invocation("int8")
        sentinel = (MagicMock(), torch.device("cuda"), lambda: None)

        with (
            patch.object(invocation, "_load_cached_encoder", return_value=sentinel) as cached,
            patch.object(Qwen2_5_VLForConditionalGeneration, "from_pretrained") as from_pretrained,
        ):
            result = invocation._load_quantized_encoder(context)

        assert result is sentinel
        cached.assert_called_once_with(context)
        from_pretrained.assert_not_called()
        context.models.make_room_in_vram.assert_not_called()


@pytest.fixture(scope="module")
def tiny_qwen_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tiny-qwen")
    torch.manual_seed(0)
    Qwen2_5_VLForConditionalGeneration(_tiny_config()).to(torch.bfloat16).save_pretrained(path, safe_serialization=True)
    return path


def _transformers_4_key(name: str) -> str:
    """The key layout the Qwen-Image `text_encoder/` folders on disk use (they predate transformers 5)."""
    for v5_prefix, v4_prefix in (("model.language_model.", "model."), ("model.visual.", "visual.")):
        if name.startswith(v5_prefix):
            return v4_prefix + name[len(v5_prefix) :]
    return name


class TestTransformersLoadPathAssumptions:
    """`_load_quantized_encoder` runs `from_pretrained` under MODEL_LOAD_LOCK's *write* lock and reads the checkpoint
    under no lock. The two facts below are why; if either test fails after a transformers bump, the lock scoping in
    `_load_quantized_encoder` has to be revisited rather than the test.
    """

    @staticmethod
    def _load(path: Path) -> Qwen2_5_VLForConditionalGeneration:
        # The exact call shape production uses (minus the device map and quantization, which need a GPU).
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            None,
            config=Qwen2_5_VLConfig.from_pretrained(str(path), local_files_only=True),
            state_dict=_read_checkpoint(path),
            dtype=torch.bfloat16,
            local_files_only=True,
        )

    def test_the_lock_is_load_bearing_a_concurrent_construction_strands_the_weights_on_meta(
        self, tiny_qwen_checkpoint: Path
    ):
        """Why the construction must exclude the cache's constructions: transformers assigns every loaded weight
        through `setattr` -> `register_parameter`, which a cache construction on another worker patches process-wide
        (`accelerate.init_empty_weights`) to route new parameters to the meta device."""
        from accelerate import init_empty_weights

        entered, release = threading.Event(), threading.Event()

        def construction_on_another_worker():
            with init_empty_weights():
                entered.set()
                release.wait()

        worker = threading.Thread(target=construction_on_another_worker)
        worker.start()
        entered.wait()
        try:
            model = self._load(tiny_qwen_checkpoint)
        finally:
            release.set()
            worker.join()

        assert any(p.device.type == "meta" for p in model.parameters()), (
            "the loaded weights no longer pass through register_parameter; revisit which lock the construction needs"
        )

    def test_the_construction_mutates_process_global_state_so_it_needs_the_write_lock(self, tiny_qwen_checkpoint: Path):
        """Why the construction takes the *write* lock rather than running as one more reader: `from_pretrained`
        builds under process-global save/restore patches - the default dtype among them - so two overlapping
        constructions would restore each other's state in the wrong order and leave a patch installed for the life
        of the process, and a concurrent VRAM move would allocate under the changed default."""
        original = torch.set_default_dtype
        set_during_load: list[torch.dtype] = []

        def spy(dtype: torch.dtype) -> None:
            set_during_load.append(dtype)
            original(dtype)

        with patch.object(torch, "set_default_dtype", spy):
            self._load(tiny_qwen_checkpoint)

        assert torch.bfloat16 in set_during_load, "the load no longer changes the process default dtype"
        assert torch.get_default_dtype() == torch.float32

    def test_transformers_4_key_layout_is_converted_on_the_state_dict_path(self, tmp_path: Path):
        """The checkpoints in production folders carry the transformers-4 names (`model.*`, `visual.*`); the
        state-dict entry point must run the same key conversion the path-based load does, or unmatched weights
        would be freshly initialised with only a load warning and the encoder would silently produce garbage."""
        torch.manual_seed(0)
        source = Qwen2_5_VLForConditionalGeneration(_tiny_config()).to(torch.bfloat16).state_dict()
        v4_names = {_transformers_4_key(name) for name in source}
        assert v4_names != set(source) and not any(name.startswith("model.language_model.") for name in v4_names)
        _tiny_config().save_pretrained(tmp_path)
        save_file(
            {_transformers_4_key(name): tensor for name, tensor in source.items()}, tmp_path / "model.safetensors"
        )

        loaded = self._load(tmp_path).state_dict()

        assert set(loaded) == set(source)
        assert all(torch.equal(loaded[name], source[name]) for name in source)


class TestQuantizedEncoderRelease:
    """The quantized encoder lives outside the cache, so `_encode` is its only owner. Its cleanup callback empties
    the CUDA cache, which only returns the ~9 GB of encoder weights to the driver if nothing still references the
    model at that point - otherwise they stay reserved by torch and the cache under-budgets the next load.
    """

    HIDDEN = 3584

    class _FakeEncoder(torch.nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            self.hidden = hidden

        def forward(self, input_ids, attention_mask, **_):
            hidden_states = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.hidden)
            return MagicMock(hidden_states=[hidden_states])

    class _ExplodingEncoder(torch.nn.Module):
        def forward(self, input_ids, attention_mask, **_):
            raise RuntimeError("CUDA out of memory (simulated)")

    def _run_encode(self, tmp_path: Path, encoder: torch.nn.Module, alive_at_cleanup: list[bool], cached: bool = False):
        """Run `_encode` as the sole owner of `encoder` (or, with `cached=True`, as a borrower of a cache-owned one),
        recording into `alive_at_cleanup` whether it was still alive when cleanup ran (recorded through the argument
        so the record survives an `_encode` that raises)."""
        model_root = tmp_path / "qwen-vl"
        (model_root / "tokenizer").mkdir(parents=True, exist_ok=True)
        context = MagicMock()
        context.models.get_absolute_path.return_value = model_root

        encoder_ref = weakref.ref(encoder)

        def cleanup():
            gc.collect()  # as production does; the frame local under test is a strong root gc cannot clear
            alive_at_cleanup.append(encoder_ref() is not None)

        seq_len = _GENERATE_DROP_IDX + 5
        model_inputs = MagicMock()
        model_inputs.input_ids = torch.zeros(1, seq_len, dtype=torch.long)
        model_inputs.attention_mask = torch.ones(1, seq_len, dtype=torch.long)
        model_inputs.to.return_value = model_inputs
        processor = MagicMock(return_value=model_inputs)

        # Hand the encoder over through a one-shot side effect: a `return_value` tuple would keep the mock holding a
        # strong reference of its own and mask the ownership being tested.
        handoff = [encoder]
        del encoder
        invocation = TestQuantizedEncoderLoad._make_invocation("none" if cached else "int8")
        loader = "_load_cached_encoder" if cached else "_load_quantized_encoder"
        with (
            patch.object(
                invocation,
                loader,
                side_effect=lambda _ctx: (handoff.pop(), torch.device("cpu"), cleanup),
            ),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()),
            patch("transformers.Qwen2_5_VLProcessor", return_value=processor),
        ):
            return invocation._encode(context, images=[])

    @staticmethod
    def _run_encoder_frame_locals(tb) -> dict:
        return next(frame.f_locals for frame, _ in traceback.walk_tb(tb) if frame.f_code.co_name == "_run_encoder")

    def test_encoder_is_released_before_cleanup_runs(self, tmp_path: Path):
        alive_at_cleanup: list[bool] = []
        prompt_embeds, mask = self._run_encode(tmp_path, self._FakeEncoder(self.HIDDEN), alive_at_cleanup)

        assert alive_at_cleanup == [False], "cleanup ran while the encoder was still referenced"
        assert prompt_embeds.shape == (1, 5, self.HIDDEN)
        assert mask is None

    def test_encoder_is_released_before_cleanup_runs_when_the_forward_raises(self, tmp_path: Path):
        """An OOM inside the forward is the likeliest failure here. The in-flight traceback holds the forward's
        frames, whose locals reference the model, so without clearing them the release is a no-op exactly when
        VRAM is scarcest - and the next generation starts from a mis-budgeted cache."""
        alive_at_cleanup: list[bool] = []
        with pytest.raises(RuntimeError, match="simulated") as excinfo:
            self._run_encode(tmp_path, self._ExplodingEncoder(), alive_at_cleanup)
        assert alive_at_cleanup == [False], "cleanup ran while the traceback still referenced the encoder"
        # The traceback is still useful for the error report: the raising line is intact, only the locals are gone.
        assert "simulated" in "".join(traceback.format_tb(excinfo.tb))
        assert "text_encoder" not in self._run_encoder_frame_locals(excinfo.tb)

    def test_forward_frames_are_cleared_on_the_cached_path_too(self, tmp_path: Path):
        """On the (default) unquantized path the cache keeps the encoder, but the forward's frames still hold its
        activations (the full-vocabulary logits and every layer's hidden states) on the GPU for as long as the
        exception lives. The error report only formats the traceback, which clearing preserves."""
        alive_at_cleanup: list[bool] = []
        with pytest.raises(RuntimeError, match="simulated") as excinfo:
            self._run_encode(tmp_path, self._ExplodingEncoder(), alive_at_cleanup, cached=True)
        assert "simulated" in "".join(traceback.format_tb(excinfo.tb))
        assert "model_inputs" not in self._run_encoder_frame_locals(excinfo.tb)
