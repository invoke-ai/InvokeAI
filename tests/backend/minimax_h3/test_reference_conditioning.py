"""Golden tests for Ref2VA reference normalization, encoding math and text presentation.

Ported behavior is asserted against upstream's own arithmetic (merged diffusers MiniMax-H3,
commit f53d552036a0d1bd5570782a39cd40cfabf112bc): the ffmpeg fps-filter resample, the
snap-down frame counts, the 2048 short-edge image rule, the truncate-then-resample audio
order, torchaudio-parity resampling (against baked torchaudio 2.11 vectors), the 2 fps
conditioner sampling, and the per-modality label presentation. No models are loaded.
"""

import warnings

import numpy as np
import pytest
import torch
from PIL import Image

from invokeai.backend.minimax_h3.keyframe_conditioning import vae_encode_autocast
from invokeai.backend.minimax_h3.reference_conditioning import (
    normalize_reference_audio,
    normalize_reference_image,
    normalize_reference_video_frames,
    normalize_reference_video_frames_streaming,
    resample_video_frame_repeats,
    resolve_reference_image_short_edge,
    sample_text_conditioning_frames,
    snap_reference_num_frames,
)
from invokeai.backend.minimax_h3.text_conditioning import MiniMaxH3TextReference, build_ref2va_presentation
from invokeai.backend.util.audio_resample import resample_sinc


class TestFrameResample:
    def test_24_to_24_is_identity(self):
        frames = np.arange(5 * 2 * 2 * 3, dtype=np.uint8).reshape(5, 2, 2, 3)
        out = normalize_reference_video_frames(frames, 24.0, 124)
        # 2x2 resolves to a 768x768 canvas, so pixels change - but the FRAME COUNT must not.
        assert out.shape[0] == 5

    def test_12_to_24_duplicates_every_frame(self):
        assert resample_video_frame_repeats(5, 12.0).tolist() == [2, 2, 2, 2, 2]

    def test_48_to_24_drops_every_other_frame(self):
        assert resample_video_frame_repeats(6, 48.0).tolist() == [1, 0, 1, 0, 1, 0]

    def test_30_to_24_drop_pattern(self):
        # ffmpeg fps-filter arithmetic: slots = floor(i * 0.8 + 0.5), repeats = diff(slots,
        # append=floor(n * 0.8 + 0.5)). Hand-computed for 10 source frames.
        assert resample_video_frame_repeats(10, 30.0).tolist() == [1, 1, 0, 1, 1, 1, 1, 0, 1, 1]

    def test_rejects_non_positive_fps(self):
        with pytest.raises(ValueError, match="positive frame rate"):
            resample_video_frame_repeats(5, 0.0)

    def test_truncates_to_generated_count(self):
        frames = np.zeros((50, 2, 2, 3), dtype=np.uint8)
        out = normalize_reference_video_frames(frames, 24.0, 22)
        assert out.shape[0] == 22

    def test_on_canvas_frames_pass_through_bitwise(self):
        rng = np.random.default_rng(0)
        frames = (rng.random((3, 768, 768, 3)) * 255).astype(np.uint8)
        out = normalize_reference_video_frames(frames, 24.0, 124)
        assert np.array_equal(out, frames)

    def test_canvas_follows_own_aspect_with_area_cap(self):
        frames = np.zeros((2, 32, 64, 3), dtype=np.uint8)  # 2:1 aspect
        out = normalize_reference_video_frames(frames, 24.0, 124)
        # resolve_canvas_size(64, 32): short edge 768 -> 1536x768, area-capped and 32-rounded.
        assert out.shape[1:3] == (704, 1440)

    def test_rejects_wrong_layout(self):
        with pytest.raises(ValueError, match="uint8 RGB frames"):
            normalize_reference_video_frames(np.zeros((5, 2, 2, 3), dtype=np.float32), 24.0, 124)

    @pytest.mark.parametrize(
        ("fps", "num_source", "num_frames"), [(30.0, 40, 22), (12.0, 10, 22), (24.0, 30, 22), (60.0, 90, 22)]
    )
    def test_streaming_equals_batch(self, fps, num_source, num_frames):
        rng = np.random.default_rng(3)
        frames = (rng.random((num_source, 32, 64, 3)) * 255).astype(np.uint8)
        batch = normalize_reference_video_frames(frames, fps, num_frames)
        streamed = normalize_reference_video_frames_streaming(iter(frames), fps, num_frames)
        assert np.array_equal(batch, streamed)

    def test_streaming_stops_early(self):
        # The iterator must not be drained past what the truncation needs.
        consumed = []

        def source():
            for index in range(1000):
                consumed.append(index)
                yield np.zeros((32, 64, 3), dtype=np.uint8)

        out = normalize_reference_video_frames_streaming(source(), 24.0, 22)
        assert out.shape[0] == 22
        assert len(consumed) == 22


class TestSnapDown:
    @pytest.mark.parametrize(("num_frames", "expected"), [(124, 124), (130, 124), (141, 141), (22, 22), (5, 22)])
    def test_snap(self, num_frames, expected):
        assert snap_reference_num_frames(num_frames) == expected


class TestImageNormalization:
    def test_upscales_to_2048_short_edge(self):
        image = Image.new("RGB", (100, 200))
        out = normalize_reference_image(image, 2048)
        # scale = 2048/100 = 20.48 -> 2048x4096, both already multiples of 32.
        assert out.size == (2048, 4096)

    def test_rounds_to_32_grid(self):
        image = Image.new("RGB", (1000, 900))
        out = normalize_reference_image(image, 2048)
        # scale = 2048/900: 1000 -> 2275.6 -> round(71.1)*32 = 2272; 900 -> 2048 exactly.
        assert out.size == (2272, 2048)

    def test_no_area_cap(self):
        out = normalize_reference_image(Image.new("RGB", (2048, 2048)), 2048)
        assert out.size == (2048, 2048)  # far above the 768x1344 canvas cap - deliberately

    def test_already_sized_image_is_returned_unresampled(self):
        image = Image.new("RGB", (2048, 2048))
        assert normalize_reference_image(image, 2048).size == (2048, 2048)

    def test_rejects_extreme_aspect(self):
        with pytest.raises(ValueError, match="1:4 and 4:1"):
            normalize_reference_image(Image.new("RGB", (100, 500)), 2048)

    def test_match_detail_targets_generation_area(self):
        # A 4000x4000 reference matched to a 768x1344 generation: short edge ~ sqrt(area).
        edge = resolve_reference_image_short_edge(4000, 4000, "match", 768 * 1344)
        assert edge == round((768 * 1344) ** 0.5)
        assert resolve_reference_image_short_edge(4000, 4000, "max", 768 * 1344) == 2048

    def test_match_never_exceeds_max(self):
        assert resolve_reference_image_short_edge(100, 100, "match", 4096 * 4096) == 2048

    def test_rejects_unknown_detail(self):
        with pytest.raises(ValueError, match="'max' or 'match'"):
            resolve_reference_image_short_edge(100, 100, "hd", 768 * 1344)


class TestAudioNormalization:
    def test_truncates_at_native_rate_before_resampling(self):
        # 2 s of 48 kHz audio truncated to 1 s (24 frames) THEN resampled: the result must
        # equal resampling the pre-truncated first second - not truncating the resampled
        # stream, which would blend post-cut samples into the tail.
        rng = np.random.default_rng(1)
        wav = (rng.random((2, 96000)) * 2 - 1).astype(np.float32)
        out = normalize_reference_audio(wav, 48000, 24)
        expected = resample_sinc(torch.from_numpy(wav[:, :48000]), 48000, 32000)
        assert torch.equal(out, expected)

    def test_mono_is_upmixed(self):
        wav = np.ones((1, 32000), dtype=np.float32)
        out = normalize_reference_audio(wav, 32000, 24)
        assert out.shape == (2, 32000)
        assert torch.equal(out[0], out[1])

    def test_native_rate_passes_through(self):
        wav = np.linspace(-1, 1, 32000, dtype=np.float32)[None].repeat(2, axis=0)
        out = normalize_reference_audio(wav, 32000, 24)
        assert torch.equal(out, torch.from_numpy(wav[:, :32000]))

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="mono or stereo"):
            normalize_reference_audio(np.zeros((3, 100), dtype=np.float32), 32000, 24)


class TestSincResampleGoldens:
    """Baked torchaudio 2.11.0 outputs (sinc_interp_hann defaults). The port must match them
    bit for bit - these vectors ARE the upstream-parity contract."""

    WAV_SEED, WAV_LEN = 7, 481

    def _wav(self):
        g = torch.Generator().manual_seed(self.WAV_SEED)
        return torch.rand(2, self.WAV_LEN, generator=g) * 2 - 1

    @pytest.mark.parametrize(
        ("orig", "new", "length", "head0", "tail1"),
        [
            (
                48,
                32,
                321,
                [
                    "-0x1.6302e00000000p-3",
                    "-0x1.e41a7c0000000p-4",
                    "0x1.0a4f960000000p-2",
                    "-0x1.ed4d160000000p-2",
                    "-0x1.b646c60000000p-3",
                    "-0x1.47e4e00000000p-4",
                    "0x1.10e8640000000p-1",
                    "0x1.06f1700000000p-1",
                ],
                ["0x1.a0e7d40000000p-1", "0x1.8aeff00000000p-2", "0x1.6e95ea0000000p-3", "0x1.c8cc3a0000000p-3"],
            ),
            (
                441,
                320,
                350,
                [
                    "-0x1.14e3f40000000p-3",
                    "-0x1.cd6f7a0000000p-3",
                    "0x1.73078e0000000p-2",
                    "-0x1.8f71f80000000p-2",
                    "-0x1.6888720000000p-2",
                    "-0x1.71f04a0000000p-4",
                    "0x1.af8ee40000000p-4",
                    "0x1.9a03440000000p-1",
                ],
                ["0x1.ee63340000000p-2", "0x1.c1e3f00000000p-4", "0x1.4f6dcc0000000p-2", "-0x1.fc556c0000000p-5"],
            ),
        ],
    )
    def test_matches_torchaudio_golden(self, orig, new, length, head0, tail1):
        out = resample_sinc(self._wav(), orig, new)
        assert out.shape == (2, length)
        assert [v.hex() for v in out[0, :8].tolist()] == head0
        assert [v.hex() for v in out[1, -4:].tolist()] == tail1

    def test_identity_rate_returns_input(self):
        wav = self._wav()
        assert resample_sinc(wav, 32000, 32000) is wav


class TestTextConditioningSampling:
    def test_samples_every_twelfth_frame(self):
        frames = np.arange(124)[:, None, None, None].astype(np.uint8) * np.ones((1, 2, 2, 3), dtype=np.uint8)
        sampled, timestamps = sample_text_conditioning_frames(frames, temporal_patch=2)
        assert [int(frame[0, 0, 0]) for frame in sampled] == list(range(0, 124, 12))
        assert len(sampled) == 11

    def test_block_timestamps_round_half_even(self):
        frames = np.zeros((124, 2, 2, 3), dtype=np.uint8)
        _, timestamps = sample_text_conditioning_frames(frames, temporal_patch=2)
        # 11 sampled frames at 0.5 s spacing, padded to 12: block means 0.25, 1.25, ... and
        # the label rendering "{:.1f}" turns 0.25 into "0.2" (round half to even).
        assert timestamps[:3] == [0.25, 1.25, 2.25]
        assert f"{timestamps[0]:.1f}" == "0.2"
        # The padded final block repeats the last timestamp.
        assert timestamps[-1] == 5.0

    def test_rejects_too_short_videos(self):
        with pytest.raises(ValueError, match="at least 13 frames"):
            sample_text_conditioning_frames(np.zeros((12, 2, 2, 3), dtype=np.uint8), temporal_patch=2)

    def test_minimum_is_temporal_patch_dependent(self):
        # temporal_patch=1 needs a single sampled frame; one source frame suffices.
        sampled, timestamps = sample_text_conditioning_frames(np.zeros((1, 2, 2, 3), dtype=np.uint8), temporal_patch=1)
        assert len(sampled) == 1 and timestamps == [0.0]


class _StubTokenizer:
    """Records every text segment; returns one distinct id per call, so the id stream maps
    1:1 onto the emission order."""

    def __init__(self):
        self.texts: list[str] = []
        self.special = {
            "<|vision_start|>": 90001,
            "<|image_pad|>": 90002,
            "<|vision_end|>": 90003,
            "<|video_pad|>": 90004,
        }

    def __call__(self, value, add_special_tokens=False):
        self.texts.append(value)
        return {"input_ids": [10000 + len(self.texts)]}

    def convert_tokens_to_ids(self, token):
        return self.special[token]


class TestPresentation:
    def test_label_order_numbering_and_tags(self):
        tokenizer = _StubTokenizer()
        references = [
            MiniMaxH3TextReference(kind="video", has_audio=True),
            MiniMaxH3TextReference(kind="image"),
            MiniMaxH3TextReference(kind="audio", has_audio=True),
        ]
        token_ids, token_tags = build_ref2va_presentation(
            tokenizer,
            "PROMPT",
            references,
            image_token_counts=[3],
            video_block_token_counts=[2],
            video_block_timestamps=[[0.25, 1.25]],
        )

        # A sounded video emits its audio label BEFORE its video label; numbering is per
        # modality; the prompt comes last.
        assert tokenizer.texts == [
            "<Audio 1>: ",
            "<Video 1>: ",
            "<0.2 seconds>",
            "<1.2 seconds>",
            "<Picture 1>: ",
            "<Audio 2>: ",
            "PROMPT",
        ]
        # Two video vision blocks (2 tokens each + start/end) and one image block (3 tokens).
        assert token_ids.count(90004) == 4
        assert token_ids.count(90002) == 3
        assert token_ids.count(90001) == token_ids.count(90003) == 3
        # Tags: labels/timestamps/prompt are text (1), vision block rows are video (0).
        assert len(token_tags) == len(token_ids)
        # Two video blocks of (start + 2 pads + end) plus one image block of (start + 3 pads + end).
        assert token_tags.count(0) == 2 * 4 + 5
        assert token_tags[-1] == 1  # the prompt

    def test_silent_video_emits_no_audio_label(self):
        tokenizer = _StubTokenizer()
        build_ref2va_presentation(
            tokenizer,
            "P",
            [MiniMaxH3TextReference(kind="video", has_audio=False)],
            image_token_counts=[],
            video_block_token_counts=[1],
            video_block_timestamps=[[0.25]],
        )
        assert tokenizer.texts == ["<Video 1>: ", "<0.2 seconds>", "P"]


def test_vae_encode_autocast_is_fp16_on_cuda_devices_and_a_no_op_elsewhere() -> None:
    with warnings.catch_warnings():
        # CPU-only torch builds warn "CUDA is not available. Disabling" on construction (and set
        # enabled=False) - the recipe's attributes are still recorded, which is what is under test.
        warnings.simplefilter("ignore")
        cuda = vae_encode_autocast(torch.device("cuda"))
    assert isinstance(cuda, torch.autocast)
    assert cuda.fast_dtype == torch.float16
    assert cuda.device == "cuda"
    # The weight-cast cache would pin float16 copies of the whole encoder for a long encode.
    assert cuda._cache_enabled is False
    for device in (torch.device("cpu"), torch.device("mps"), torch.device("meta")):
        with vae_encode_autocast(device):
            assert not torch.is_autocast_enabled("cpu")
