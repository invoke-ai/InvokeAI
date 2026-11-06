"""Third-party PIL custom nodes keep working through the real invocation image interface.

The fixture in `third_party_cinematic_effects/` is six unmodified nodes of JPPhoto's cinematic-effects pack (see its
PROVENANCE.md). They are registered the way the custom-node loader registers a pack, then run through real image
services over a temporary SQLite database and temporary disk storage: `context.images.get_pil` -> PIL work ->
`context.images.save` -> `ImageOutput`. Nothing is downloaded; Lens Blur gets a depth map and an aperture image
supplied, so its depth model is never requested.
"""

import importlib.util
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageChops, ImageDraw

from invokeai.app.invocations.baseinvocation import BaseInvocation, InvocationRegistry
from invokeai.app.invocations.fields import ImageField
from invokeai.app.invocations.image import BlankImageInvocation
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.image_files.image_files_disk import DiskImageFileStorage
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.names.names_default import SimpleNameService
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItem
from invokeai.app.services.shared.graph import Graph, GraphExecutionState
from invokeai.app.services.shared.invocation_context import (
    InvocationContext,
    InvocationContextData,
    build_invocation_context,
)
from invokeai.app.services.urls.urls_default import LocalUrlService

PACK_DIR = Path(__file__).parent / "third_party_cinematic_effects" / "cinematic_effects"
# The pack is imported under a name of its own: the first segment of a module name is the node pack name.
PACK_NAME = "jpphoto_cinematic_effects_fixture"
# module -> (invocation type, class name)
NODE_MODULES = {
    "chromatic_aberration": ("chromatic_aberration", "ChromaticAberrationInvocation"),
    "lens_vignette": ("lens_vignette", "LensVignetteInvocation"),
    "lens_blur": ("lens_blur", "LensBlurInvocation"),
    "curves": ("curves_invocation", "CurvesInvocation"),
    "film_grain": ("film_grain", "FilmGrainInvocation"),
    "spherical_distortion": ("spherical_distortion", "SphericalDistortionInvocation"),
}
SIZE = (64, 48)


@pytest.fixture(scope="module")
def pack() -> Iterator[dict[str, type[BaseInvocation]]]:
    """Loads the fixture nodes as the custom-node loader loads a pack, and removes them again afterwards."""
    registered_before = set(InvocationRegistry.get_invocation_types())

    package = types.ModuleType(PACK_NAME)
    package.__path__ = [str(PACK_DIR)]
    sys.modules[PACK_NAME] = package
    try:
        # util is imported by relative path from the nodes
        for name in ("util", *NODE_MODULES):
            spec = importlib.util.spec_from_file_location(f"{PACK_NAME}.{name}", PACK_DIR / f"{name}.py")
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        # film_grain.py registers more than one type, so look at what the pack actually registered.
        loaded = {
            c.get_type() for c in InvocationRegistry.get_invocation_classes() if c.UIConfig.node_pack == PACK_NAME
        }
        assert not loaded & registered_before, "a fixture node type replaced an already registered node"
        yield {
            invocation_type: getattr(sys.modules[f"{PACK_NAME}.{module}"], class_name)
            for module, (invocation_type, class_name) in NODE_MODULES.items()
        }
    finally:
        InvocationRegistry.unregister_pack(PACK_NAME)
        for name in [name for name in sys.modules if name == PACK_NAME or name.startswith(f"{PACK_NAME}.")]:
            del sys.modules[name]


@pytest.fixture
def invoker(mock_invoker: Invoker, tmp_path: Path) -> Invoker:
    """Real image records (in-memory SQLite), real disk storage under tmp_path, real image service."""
    services = mock_invoker.services
    services.image_files = DiskImageFileStorage(tmp_path / "images")
    services.names = SimpleNameService()
    services.urls = LocalUrlService()
    services.image_files.start(mock_invoker)
    services.images.start(mock_invoker)
    return mock_invoker


def _source_image(mode: str) -> Image.Image:
    """Coloured blocks over a gradient, with a varying alpha: every node has something to change."""
    width, height = SIZE
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :].repeat(height, axis=0)
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None].repeat(width, axis=1)
    rgb = Image.fromarray(np.stack([x, y, 255 - x], axis=2), "RGB")
    draw = ImageDraw.Draw(rgb)
    for left in range(0, width, 16):
        draw.rectangle((left, 8, left + 7, 39), fill=(255, 255, 255) if left % 32 == 0 else (10, 10, 10))
    image = rgb.convert("RGBA")
    image.putalpha(Image.fromarray(np.linspace(100, 255, width, dtype=np.uint8)[None, :].repeat(height, axis=0)))
    return image.convert(mode)


def _store(invoker: Invoker, image: Image.Image) -> str:
    return invoker.services.images.create(
        image=image, image_origin=ResourceOrigin.INTERNAL, image_category=ImageCategory.GENERAL
    ).image_name


def _context_for(invoker: Invoker, node: BaseInvocation) -> tuple[BaseInvocation, InvocationContext]:
    """The prepared invocation and real context the session runner would hand to `invoke`."""
    graph = Graph()
    graph.add_node(node)
    session = GraphExecutionState(graph=graph)
    queue_item = SessionQueueItem(
        item_id=1,
        batch_id="batch",
        session_id=session.id,
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
        started_at=None,
        completed_at=None,
        queue_id="default",
        session=session,
    )
    invocation = session.next()
    assert invocation is not None
    context = build_invocation_context(
        services=invoker.services,
        data=InvocationContextData(queue_item=queue_item, invocation=invocation, source_invocation_id=node.id),
        is_canceled=lambda: False,
    )
    return invocation, context


def _run(invoker: Invoker, node: BaseInvocation) -> ImageOutput:
    invocation, context = _context_for(invoker, node)
    output = invocation.invoke(context)
    assert isinstance(output, ImageOutput)
    return output


def _lens_blur_inputs(invoker: Invoker) -> dict[str, Any]:
    depth = Image.fromarray(np.linspace(0, 255, SIZE[0], dtype=np.uint8)[None, :].repeat(SIZE[1], axis=0), "L")
    aperture = Image.new("L", (512, 512), 0)
    ImageDraw.Draw(aperture).ellipse((56, 56, 456, 456), fill=255)
    return {
        "depth_map": ImageField(image_name=_store(invoker, depth)),
        # Optional in the declaration, but the node dereferences it unconditionally.
        "aperture_image": ImageField(image_name=_store(invoker, aperture)),
        "focal_distance": 0.0,
        "distance_range": 0.2,
        "max_blur_radius": 4.0,
        "max_blur_steps": 4,
        "reduce_halos": False,
    }


# (node type, input mode, output mode, alpha, extra fields). Alpha is "kept" (identical to the source), "opaque"
# (discarded: Lens Blur and Film Grain convert their RGB result back to the input mode, which makes it fully opaque;
# that is how the pack behaves today and is recorded as such) or "warped" (moved along with the pixels).
# Curves widens L to RGB and LA to RGBA.
CURVES = {"mapping_str": "0:0,128:40,255:255"}
CASES = [
    ("chromatic_aberration", "RGB", "RGB", None, {"strength": 6.0}),
    ("chromatic_aberration", "RGBA", "RGBA", "kept", {"strength": 6.0}),
    ("lens_vignette", "RGB", "RGB", None, {"intensity": 2.0}),
    ("lens_vignette", "RGBA", "RGBA", "kept", {"intensity": 2.0}),
    ("lens_blur", "RGB", "RGB", None, {}),
    ("lens_blur", "RGBA", "RGBA", "opaque", {}),
    ("curves_invocation", "RGB", "RGB", None, CURVES),
    ("curves_invocation", "RGBA", "RGBA", "kept", CURVES),
    ("curves_invocation", "L", "RGB", None, CURVES),
    ("curves_invocation", "LA", "RGBA", "kept", CURVES),
    ("film_grain", "RGB", "RGB", None, {"seed_1": 1, "seed_2": 2}),
    ("film_grain", "RGBA", "RGBA", "opaque", {"seed_1": 1, "seed_2": 2}),
    ("spherical_distortion", "RGB", "RGB", None, {"k1": 0.4}),
    ("spherical_distortion", "RGBA", "RGBA", "warped", {"k1": 0.4}),
]


def test_all_six_nodes_register(pack: dict[str, type[BaseInvocation]]) -> None:
    for invocation_type, node_class in pack.items():
        assert InvocationRegistry.get_invocation_for_type(invocation_type) is node_class
        assert node_class.UIConfig.node_pack == PACK_NAME


@pytest.mark.parametrize(("invocation_type", "mode", "result_mode", "alpha_outcome", "fields"), CASES)
def test_node_saves_a_readable_changed_image(
    pack: dict[str, type[BaseInvocation]],
    invoker: Invoker,
    invocation_type: str,
    mode: str,
    result_mode: str,
    alpha_outcome: str | None,
    fields: dict[str, Any],
) -> None:
    source = _source_image(mode)
    source_name = _store(invoker, source)
    stored_source = invoker.services.images.get_pil_image(source_name)
    if invocation_type == "lens_blur":
        fields = {**fields, **_lens_blur_inputs(invoker)}

    node = pack[invocation_type](id="node", image=ImageField(image_name=source_name), **fields)
    output = _run(invoker, node)

    result = invoker.services.images.get_pil_image(output.image.image_name)
    assert invoker.services.images.get_dto(output.image.image_name).image_name == output.image.image_name
    assert (output.width, output.height) == result.size
    assert result.mode == result_mode
    if invocation_type != "spherical_distortion":
        # Spherical Distortion crops to the distorted content; the others keep the frame.
        assert result.size == SIZE
    # The effect moved colour/luma pixels (alpha is compared separately).
    colour_before, colour_after = stored_source.convert("RGB"), result.convert("RGB")
    assert colour_before.size != colour_after.size or ImageChops.difference(colour_before, colour_after).getbbox()

    if alpha_outcome is not None:
        alpha = result.getchannel("A")
        if alpha_outcome == "kept":
            assert ImageChops.difference(alpha, stored_source.getchannel("A")).getbbox() is None
        elif alpha_outcome == "opaque":
            assert alpha.getextrema() == (255, 255)
        else:
            assert alpha.getextrema()[0] < 255


def test_lens_blur_aperture_is_required_in_practice(pack: dict[str, type[BaseInvocation]], invoker: Invoker) -> None:
    """Baseline: `aperture_image` is declared optional, yet the node reads it unconditionally."""
    fields = _lens_blur_inputs(invoker)
    del fields["aperture_image"]
    node = pack["lens_blur"](id="node", image=ImageField(image_name=_store(invoker, _source_image("RGB"))), **fields)

    with pytest.raises(AttributeError):
        _run(invoker, node)


def test_get_pil_results_are_independent_of_the_service_cache(invoker: Invoker) -> None:
    """Whatever a node does to the image `get_pil` hands out, the next read sees the stored pixels."""
    name = _store(invoker, _source_image("RGB"))
    expected = invoker.services.images.get_pil_image(name).tobytes()
    _, context = _context_for(invoker, BlankImageInvocation(id="node"))

    for mode in (None, "RGB", "L"):
        mutated = context.images.get_pil(name, mode)
        mutated.paste((1, 2, 3) if mutated.mode == "RGB" else 7, (0, 0, *SIZE))
        assert context.images.get_pil(name).tobytes() == expected
        assert invoker.services.images.get_pil_image(name).tobytes() == expected
