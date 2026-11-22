"""Validation-level tests for the MiniMax H3 Ref2VA reference invocations.

No models are loaded: these exercise field construction, kind/cap validation, the
prompt-vs-conditioning signature cross-checks in the denoise node, and the task-variant
guard. The numerics live in tests/backend/minimax_h3/.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from invokeai.app.invocations.fields import (
    ImageField,
    MiniMaxH3ConditioningField,
    MiniMaxH3EncodedReferenceField,
    MiniMaxH3FrameConditioningField,
    MiniMaxH3ReferenceConditioningField,
    MiniMaxH3ReferenceMediaField,
    VideoField,
)
from invokeai.app.invocations.minimax_h3.minimax_h3_denoise import MiniMaxH3DenoiseInvocation
from invokeai.app.invocations.minimax_h3.minimax_h3_reference import (
    _ResolvedVideoRange,
    load_reference_audio,
    normalize_reference_list,
    reference_has_audio,
    reference_kind,
    reference_signature_entry,
)
from invokeai.app.invocations.model import MiniMaxH3TransformerField, ModelIdentifierField
from invokeai.app.invocations.text_encoder.minimax_h3_text_encoder import MiniMaxH3TextEncoderInvocation
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import MiniMaxH3ConditioningInfo


def _image_ref(name: str = "img.png", detail: str = "max") -> MiniMaxH3ReferenceMediaField:
    return MiniMaxH3ReferenceMediaField(image=ImageField(image_name=name), image_detail=detail)


def _video_ref(name: str = "clip.mp4", conditioning: str = "video_audio") -> MiniMaxH3ReferenceMediaField:
    return MiniMaxH3ReferenceMediaField(video=VideoField(video_name=name), video_conditioning=conditioning)


class TestMediaField:
    def test_kind_mapping(self):
        assert reference_kind(_image_ref()) == "image"
        assert reference_kind(_video_ref()) == "video"
        assert reference_kind(_video_ref(conditioning="video")) == "video"
        assert reference_kind(_video_ref(conditioning="audio")) == "audio"

    def test_has_audio(self):
        assert not reference_has_audio(_image_ref())
        assert reference_has_audio(_video_ref())
        assert not reference_has_audio(_video_ref(conditioning="video"))
        assert reference_has_audio(_video_ref(conditioning="audio"))

    def test_normalize_rejects_both_or_neither_media(self):
        with pytest.raises(ValueError, match="exactly one"):
            normalize_reference_list([MiniMaxH3ReferenceMediaField()])
        with pytest.raises(ValueError, match="exactly one"):
            normalize_reference_list(
                [MiniMaxH3ReferenceMediaField(image=ImageField(image_name="a"), video=VideoField(video_name="b"))]
            )

    def test_normalize_accepts_all_audio(self):
        references = [_video_ref(conditioning="audio")]
        assert normalize_reference_list(references) == references

    def test_normalize_rejects_caps(self):
        with pytest.raises(ValueError, match="At most 3 video"):
            normalize_reference_list([_video_ref(name=f"v{i}.mp4") for i in range(4)])
        with pytest.raises(ValueError, match="At most 9 image"):
            normalize_reference_list([_image_ref(name=f"i{i}.png") for i in range(10)])

    def test_normalize_wraps_single_reference(self):
        assert len(normalize_reference_list(_video_ref())) == 1

    def test_signature_distinguishes_options_and_order_inputs(self):
        base = reference_signature_entry(_video_ref(), None, 124)
        assert reference_signature_entry(_video_ref(conditioning="video"), None, 124) != base
        assert reference_signature_entry(_video_ref(name="other.mp4"), None, 124) != base
        assert reference_signature_entry(_video_ref(), None, 141) != base
        trimmed = _video_ref()
        trimmed.start_frame = 10
        assert reference_signature_entry(trimmed, None, 124) != base
        image = reference_signature_entry(_image_ref(), (2048, 2048), 124)
        assert reference_signature_entry(_image_ref(detail="match"), (2048, 2048), 124) != image
        assert reference_signature_entry(_image_ref(), (1024, 1024), 124) != image


class TestResolvedVideoRange:
    """The reference range's start/length enforcement. No decoding: the probe is stubbed."""

    @staticmethod
    def _resolve(start: int, end: int, n_frames: int = 300) -> _ResolvedVideoRange:
        context = MagicMock()
        context.videos.get_path.return_value = "clip.mp4"
        reference = _video_ref()
        reference.start_frame = start
        reference.end_frame = end
        with (
            patch(
                "invokeai.app.invocations.minimax_h3.minimax_h3_reference.probe_video",
                return_value=(640, 480, n_frames / 24.0, 24.0),
            ),
            patch(
                "invokeai.app.invocations.minimax_h3.minimax_h3_reference.decoder_frame_count", return_value=n_frames
            ),
        ):
            return _ResolvedVideoRange(context, reference)

    def test_resolves_positive_and_negative_bounds(self):
        span = self._resolve(10, 49)
        assert (span.start, span.end) == (10, 49)
        tail = self._resolve(-24, -1)
        assert (tail.start, tail.end) == (276, 299)

    def test_pins_the_length_to_the_last_frame(self):
        # A start frame plus a length that runs past the end takes the rest of the clip
        # rather than failing the generation.
        span = self._resolve(250, 449)
        assert (span.start, span.end) == (250, 299)

    def test_rejects_a_start_frame_past_the_end(self):
        with pytest.raises(ValueError, match="start_frame=300 is out of range"):
            self._resolve(300, 400)
        with pytest.raises(ValueError, match="start_frame=-301 is out of range"):
            self._resolve(-301, -1)

    def test_rejects_an_inverted_range(self):
        with pytest.raises(ValueError, match="must not be before start_frame"):
            self._resolve(100, 50)


def _transformer_field(variant: str | None) -> MiniMaxH3TransformerField:
    return MiniMaxH3TransformerField(
        transformer=ModelIdentifierField(
            key="t", hash="h", name="T", base=BaseModelType.MiniMaxH3, type=ModelType.Main
        ),
        variant=variant,
    )


def _reference_conditioning(signature: list[str], num_frames: int = 124) -> MiniMaxH3ReferenceConditioningField:
    return MiniMaxH3ReferenceConditioningField(
        references=[MiniMaxH3EncodedReferenceField(kind="image", video_rows_name="rows")],
        num_frames=num_frames,
        signature=signature,
    )


def _denoise(reference_conditioning=None, variant="ref2va", num_frames="124") -> MiniMaxH3DenoiseInvocation:
    return MiniMaxH3DenoiseInvocation(
        id="denoise",
        transformer=_transformer_field(variant),
        positive_conditioning=MiniMaxH3ConditioningField(conditioning_name="cond"),
        reference_conditioning=reference_conditioning,
        num_frames=num_frames,
    )


def _denoise_with_frames(reference_conditioning=None, variant="ref2va") -> MiniMaxH3DenoiseInvocation:
    node = _denoise(reference_conditioning, variant=variant)
    node.frame_conditioning = MiniMaxH3FrameConditioningField(
        condition_rows_name="rows", keyframe_anchors=["first", "last"], width=1344, height=768
    )
    return node


def _context_with_conditioning(cond_info: MiniMaxH3ConditioningInfo) -> MagicMock:
    context = MagicMock()
    cond_data = MagicMock()
    cond_data.conditionings = [cond_info]
    context.conditioning.load.return_value = cond_data
    return context


def _cond_info(
    reference_signature=(), reference_num_frames=None, keyframe_anchors=(), canvas=(1344, 768)
) -> MiniMaxH3ConditioningInfo:
    return MiniMaxH3ConditioningInfo(
        prompt_embeds=torch.zeros(1, 3, 8),
        text_token_tags=torch.tensor([1, 1, 0]),
        reference_signature=reference_signature,
        reference_num_frames=reference_num_frames,
        keyframe_anchors=keyframe_anchors,
        width=canvas[0] if keyframe_anchors else None,
        height=canvas[1] if keyframe_anchors else None,
    )


class TestDenoiseGuards:
    def test_rejects_references_on_fl2va_transformer(self):
        node = _denoise(_reference_conditioning(["image:a:max:64x64"]), variant="fl2va")
        with pytest.raises(ValueError, match="FL2VA task checkpoint"):
            node.invoke(_context_with_conditioning(_cond_info()))

    def test_rejects_ref2va_transformer_without_references(self):
        node = _denoise(None, variant="ref2va")
        with pytest.raises(ValueError, match="requires reference conditioning"):
            node.invoke(_context_with_conditioning(_cond_info()))

    def test_rejects_prompt_references_without_wired_conditioning(self):
        node = _denoise(None, variant=None)
        with pytest.raises(ValueError, match="no reference conditioning is wired"):
            node.invoke(_context_with_conditioning(_cond_info(reference_signature=("image:a:max:64x64",))))

    def test_wired_conditioning_with_referenceless_prompt_names_the_right_remedy(self):
        # The prompt side has NO references: the error must say to wire them there, not to
        # change a frame count (the num_frames comparison would otherwise fire first).
        node = _denoise(_reference_conditioning(["image:a:max:64x64"]), variant=None)
        with pytest.raises(ValueError, match="Connect the same ordered references to Prompt"):
            node.invoke(_context_with_conditioning(_cond_info()))

    def test_rejects_canvas_mismatch(self):
        sig = "image:a:max:64x64"
        field = _reference_conditioning([sig])
        field.width = 768
        field.height = 768
        node = _denoise(field, variant=None)
        cond = _cond_info(reference_signature=(sig,), reference_num_frames=124)
        with pytest.raises(ValueError, match="768x768 canvas"):
            node.invoke(_context_with_conditioning(cond))

    def test_rejects_signature_mismatch(self):
        sig = "image:a:max:64x64"
        node = _denoise(_reference_conditioning([sig]), variant=None)
        cond = _cond_info(reference_signature=("image:OTHER:max:64x64",), reference_num_frames=124)
        with pytest.raises(ValueError, match="Reference mismatch"):
            node.invoke(_context_with_conditioning(cond))

    def test_rejects_num_frames_mismatch_between_nodes(self):
        sig = "image:a:max:64x64"
        node = _denoise(_reference_conditioning([sig], num_frames=141), variant=None)
        cond = _cond_info(reference_signature=(sig,), reference_num_frames=141)
        with pytest.raises(ValueError, match="prepared for 141 frames"):
            node.invoke(_context_with_conditioning(cond))

    def test_rejects_frames_with_references_on_fl2va_transformer(self):
        node = _denoise_with_frames(_reference_conditioning(["image:a:max:64x64"]), variant="fl2va")
        with pytest.raises(ValueError, match="FL2VA task checkpoint"):
            node.invoke(_context_with_conditioning(_cond_info()))

    def test_ref2va_transformer_still_requires_references_when_frames_are_wired(self):
        node = _denoise_with_frames(None, variant="ref2va")
        with pytest.raises(ValueError, match="requires reference conditioning"):
            node.invoke(_context_with_conditioning(_cond_info(keyframe_anchors=("first", "last"))))

    def test_frames_with_references_checks_prompt_keyframes(self):
        sig = "image:a:max:64x64"
        node = _denoise_with_frames(_reference_conditioning([sig]))
        cond = _cond_info(reference_signature=(sig,), reference_num_frames=124)
        with pytest.raises(ValueError, match="Keyframe mismatch"):
            node.invoke(_context_with_conditioning(cond))

    @pytest.mark.parametrize("prompt_anchors", [("first",), ("last",), ("last", "first")])
    def test_frames_with_references_rejects_prompt_anchors_that_differ_in_subset_or_order(self, prompt_anchors):
        sig = "image:a:max:64x64"
        node = _denoise_with_frames(_reference_conditioning([sig]))
        cond = _cond_info(reference_signature=(sig,), reference_num_frames=124, keyframe_anchors=prompt_anchors)
        with pytest.raises(ValueError, match="Keyframe mismatch"):
            node.invoke(_context_with_conditioning(cond))

    def test_frames_with_references_checks_prompt_reference_signature(self):
        node = _denoise_with_frames(_reference_conditioning(["image:a:max:64x64"]))
        cond = _cond_info(
            reference_signature=("image:OTHER:max:64x64",),
            reference_num_frames=124,
            keyframe_anchors=("first", "last"),
        )
        with pytest.raises(ValueError, match="Reference mismatch"):
            node.invoke(_context_with_conditioning(cond))

    def test_frames_with_references_checks_prompt_keyframe_canvas(self):
        sig = "image:a:max:64x64"
        node = _denoise_with_frames(_reference_conditioning([sig]))
        cond = _cond_info(
            reference_signature=(sig,), reference_num_frames=124, keyframe_anchors=("first", "last"), canvas=(768, 768)
        )
        with pytest.raises(ValueError, match="keyframes were prepared at 768x768"):
            node.invoke(_context_with_conditioning(cond))

    @pytest.mark.parametrize("anchors", [("first",), ("last",), ("first", "last")])
    def test_builds_one_packed_state_with_keyframes_ahead_of_references(self, anchors):
        sig = "image:a:max:1x6x2"
        node = _denoise_with_frames(_reference_conditioning([sig]))
        node.steps = 2
        node.reference_conditioning.references = [
            MiniMaxH3EncodedReferenceField(
                kind="image", video_rows_name="ref_rows", latent_frames=1, latent_height=6, latent_width=2
            )
        ]
        cond = _cond_info(reference_signature=(sig,), reference_num_frames=124, keyframe_anchors=anchors)
        # Fixture sizes, stated outright: a 768x1344 canvas is a 48x84 latent grid with 2x2 patches
        # (24 x 42 = 1008 rows per latent frame); 124 frames are 37 latent frames; the reference is
        # a 1x6x2 latent image (3 rows x 96 channels).
        rows_per_frame, target_frames, reference_rows = 1008, 37, 3
        keyframe_rows = len(anchors) * rows_per_frame
        text_rows = cond.text_token_tags.shape[0]
        context = MagicMock()
        context.tensors.load.return_value = torch.zeros(reference_rows, 96)

        state = node._build_reference_state(
            context,
            cond,
            124,
            target_frames,
            48,
            84,
            200,
            torch.device("cpu"),
            anchors,
            torch.zeros(keyframe_rows, 96),
        )

        conditioning = keyframe_rows + reference_rows
        assert state.layout.num_condition_video_rows == conditioning
        assert state.video_rows.shape[0] == conditioning + target_frames * rows_per_frame
        # Packed order: text, keyframes, the reference, then everything generated.
        assert state.layout.video_indices[:conditioning].tolist() == list(range(text_rows, text_rows + conditioning))

    def test_rejects_still_image_frame_count_with_references(self):
        sig = "image:a:max:64x64"
        node = _denoise(_reference_conditioning([sig], num_frames=5), variant=None, num_frames="5")
        cond = _cond_info(reference_signature=(sig,), reference_num_frames=5)
        with pytest.raises(ValueError, match="still-image"):
            node.invoke(_context_with_conditioning(cond))


class TestTextEncoderGuards:
    def _node(self, **kwargs) -> MiniMaxH3TextEncoderInvocation:
        identifier = ModelIdentifierField(
            key="te", hash="h", name="TE", base=BaseModelType.MiniMaxH3, type=ModelType.Main
        )
        from invokeai.app.invocations.model import MiniMaxH3TextEncoderField

        return MiniMaxH3TextEncoderInvocation(
            id="prompt",
            prompt="p",
            text_encoder=MiniMaxH3TextEncoderField(tokenizer=identifier, processor=identifier, text_encoder=identifier),
            **kwargs,
        )

    def _encode(self, node, images):
        """Run the node with a stubbed model stack; returns (ref2va encoder kwargs, saved conditioning info)."""
        context = MagicMock()
        context.images.get_pil.side_effect = lambda name: images[name]
        context.models.load.return_value.model_on_device.return_value.__enter__.return_value = (None, MagicMock())
        saved = []
        context.conditioning.save.side_effect = lambda data: saved.append(data) or "cond"
        module = "invokeai.app.invocations.text_encoder.minimax_h3_text_encoder"
        with (
            patch(
                f"{module}.encode_prompt_ref2va", return_value=(torch.zeros(1, 3, 8), torch.tensor([1, 0, 0]))
            ) as ref,
            patch(f"{module}.get_effective_device", return_value=torch.device("cpu")),
        ):
            node.invoke(context)
        return ref.call_args.kwargs, saved[0].conditionings[0]

    @pytest.mark.parametrize(
        ("keyframes", "anchors"),
        [
            ({"first_image": ImageField(image_name="kf1")}, ("first",)),
            ({"last_image": ImageField(image_name="kf2")}, ("last",)),
            (
                {"first_image": ImageField(image_name="kf1"), "last_image": ImageField(image_name="kf2")},
                ("first", "last"),
            ),
        ],
    )
    def test_keyframes_with_references_encode_only_the_references_and_record_the_anchors(self, keyframes, anchors):
        node = self._node(references=_image_ref("ref"), num_frames=124, **keyframes)
        images = {
            "kf1": Image.new("RGB", (40, 40), (255, 0, 0)),
            "kf2": Image.new("RGB", (40, 40), (0, 255, 0)),
            "ref": Image.new("RGB", (64, 64), (0, 0, 255)),
        }

        kwargs, info = self._encode(node, images)

        (reference,) = kwargs["references"]
        assert reference.kind == "image"
        assert reference.image.getpixel((reference.image.width // 2, reference.image.height // 2)) == (0, 0, 255)
        assert "keyframe_images" not in kwargs
        assert info.keyframe_anchors == anchors
        assert (info.width, info.height) == (node.width, node.height)
        assert len(info.reference_signature) == 1
        assert info.reference_num_frames == 124

    def test_rejects_an_empty_reference_list_with_keyframes(self):
        # The empty list is rejected like everywhere else; it must not fall back to a prompt that
        # silently drops the keyframe vision context.
        node = self._node(first_image=ImageField(image_name="kf1"), references=[], num_frames=124)
        with pytest.raises(ValueError, match="at least one reference"):
            self._encode(node, {"kf1": Image.new("RGB", (40, 40))})

    def test_requires_num_frames_with_references(self):
        node = self._node(references=_image_ref())
        with pytest.raises(ValueError, match="num_frames is required"):
            node.invoke(MagicMock())


class _Span:
    """The parts of `_ResolvedVideoRange` that `load_reference_audio` reads."""

    def __init__(self, start: int, end: int, fps: float, path: str = "clip.mp4") -> None:
        self.start = start
        self.end = end
        self.fps = fps
        self.path = path


class TestReferenceAudioWindow:
    """The soundtrack window is bounded by the GENERATED duration before anything copies it.

    An audio-only reference defaults to its whole clip (the frontend's
    `getDefaultReferenceClip`), which for an uploaded song is minutes of PCM. Only
    `num_frames / 24` seconds of it ever survive normalization, so the window has to be
    clamped before the silence-pad -- `np.pad` allocates the whole window, however few
    samples it actually adds.
    """

    RATE = 32_000  # the audio VAE's own rate, so normalization is a pass-through
    FPS = 24.0  # AUDIO_WRAP_FPS: what the ingest wrapper renders an audio upload at

    def _patch_pcm(self, monkeypatch, pcm):
        monkeypatch.setattr(
            "invokeai.app.invocations.minimax_h3.minimax_h3_reference.extract_audio_pcm",
            lambda path, float_pcm=False: (pcm, self.RATE),
        )

    def test_whole_clip_window_keeps_only_the_generated_duration(self, monkeypatch):
        # A 60 s track behind a 14.375 s (345-frame) generation.
        pcm = np.linspace(-1, 1, 60 * self.RATE, dtype=np.float32)[None].repeat(2, axis=0)
        self._patch_pcm(monkeypatch, pcm)
        out = load_reference_audio(_video_ref(conditioning="audio"), _Span(0, 60 * 24 - 1, self.FPS), 345)

        kept = int(345 / self.FPS * self.RATE)
        assert out.shape == (2, kept)
        assert torch.equal(out, torch.from_numpy(pcm[:, :kept]))

    def test_short_track_pads_to_the_generated_duration_not_the_clip(self, monkeypatch):
        # The track ends long before the video does: the silence-pad must fill the
        # generated duration, NOT the 60 s the whole-clip window nominally spans.
        pcm = np.ones((2, 5 * self.RATE), dtype=np.float32)
        self._patch_pcm(monkeypatch, pcm)
        out = load_reference_audio(_video_ref(conditioning="audio"), _Span(0, 60 * 24 - 1, self.FPS), 345)

        kept = int(345 / self.FPS * self.RATE)
        assert out.shape == (2, kept)
        assert torch.equal(out[:, : 5 * self.RATE], torch.ones(2, 5 * self.RATE))
        assert torch.equal(out[:, 5 * self.RATE :], torch.zeros(2, kept - 5 * self.RATE))

    def test_does_not_pad_beyond_the_generated_duration(self, monkeypatch):
        # The actual regression guard. The two tests above only prove the OUTPUT is
        # unchanged -- normalization truncates either way, so they pass with or without the
        # clamp. This one watches the allocation: unclamped, np.pad is asked to build the
        # whole 60 s window (a full-length copy of an uploaded song) to preserve an
        # alignment that is discarded on the next line.
        pcm = np.ones((2, 5 * self.RATE), dtype=np.float32)
        self._patch_pcm(monkeypatch, pcm)
        widths: list[int] = []
        real_pad = np.pad

        def spy(array, pad_width, *args, **kwargs):
            widths.append(array.shape[1] + pad_width[1][1])

            return real_pad(array, pad_width, *args, **kwargs)

        monkeypatch.setattr(np, "pad", spy)
        load_reference_audio(_video_ref(conditioning="audio"), _Span(0, 60 * 24 - 1, self.FPS), 345)

        assert widths == [int(345 / self.FPS * self.RATE)]

    def test_window_shorter_than_the_generation_is_left_alone(self, monkeypatch):
        # The clamp must not lengthen a deliberately short window: 200 frames of a
        # 345-frame generation still yields 200 frames' worth of audio, cut short.
        pcm = np.linspace(-1, 1, 60 * self.RATE, dtype=np.float32)[None].repeat(2, axis=0)
        self._patch_pcm(monkeypatch, pcm)
        out = load_reference_audio(_video_ref(conditioning="audio"), _Span(0, 199, self.FPS), 345)

        assert out.shape == (2, round(200 * self.RATE / self.FPS))
