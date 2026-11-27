"""Scaffold the files a new model architecture needs, and list the ones it cannot scaffold.

    python scripts/new_architecture.py --base new-model --name NewModel        # dry run
    python scripts/new_architecture.py --base new-model --name NewModel --write

Three files can be generated, because the registry made them mechanical: the architecture's
declaration under `architectures/defs/`, its invocation package, and its starter-model module. The
declaration is generated with every required facet present but obviously wrong, so it fails loudly
at boot until someone fills it in — a stub that boots would be worse than no stub.

"Fails loudly" is carried by `LatentSpace.__post_init__`, not by `registry.validate()`, which only
checks that the required facets are *present*: the placeholder projection is a single all-zero row,
which no VAE has, so constructing it raises while `defs/` is being imported. That is before the
registry is even called, and the traceback names the generated file.

The rest cannot be generated, and the point of this script is as much to enumerate that rest as to
write the three. That list is *derived* on each run rather than written down here: any module naming
five or more `BaseModelType` members is dispatching on base, so a new one has to be added to it by
hand. A hardcoded list would be wrong the first time someone removed a dispatch — which is what the
last several changes have been doing.
"""

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "invokeai"

# Excluded from the derived list: the registry is the mechanism, not the cost, and the starter
# package is scaffolded below. Shipped migrations name many bases but must never be edited — they
# record what already-migrated databases were told they got.
NOT_A_COST = (
    "invokeai/frontend",
    "invokeai/backend/architectures/",
    "invokeai/backend/model_manager/starter_models/",
    "invokeai/app/services/shared/sqlite_migrator/migrations/",
)

# Files a derivation cannot find, because the work there is to *add* something rather than to extend
# an existing dispatch. Kept short and specific; each says what to do, not just where.
UNDERIVABLE = [
    (
        "invokeai/backend/model_manager/taxonomy.py",
        "Add the `BaseModelType` member. If the architecture has variants, add its enum and list it "
        "in `AnyVariant` and `variant_type_adapter`.",
    ),
    (
        "invokeai/backend/stable_diffusion/diffusion/conditioning_data.py",
        "Define the `*ConditioningInfo` its text encoder produces, unless it reuses an existing one.",
    ),
    (
        "invokeai/app/invocations/metadata.py",
        "Add the mode strings to `GENERATION_MODES`. They must match what the new `ModalityFacet` "
        "declares — a test compares the two — and they are persisted in image metadata, so they "
        "cannot be changed later.",
    ),
]


def defs_module(base_value: str, enum_name: str) -> str:
    """The architecture's declaration. Every required facet, none of them plausible."""
    return f'''"""What the {base_value} architecture declares."""

from invokeai.backend.architectures.facets.conditioning import ConditioningFacet
from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
from invokeai.backend.architectures.facets.features import FeaturesFacet, NegativePrompt
from invokeai.backend.architectures.facets.latent_space import LatentSpace, LatentSpaceFacet
from invokeai.backend.architectures.facets.modality import ModalityFacet
from invokeai.backend.architectures.registry import register
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import BasicConditioningInfo

# TODO: every value below is a placeholder. The registry refuses to boot without these facets, but
# it cannot tell whether they are true — replace each one before generating anything.

# TODO: the latent -> RGB projection for this VAE, one row per latent channel. If the architecture
# reuses another's VAE, import that LatentSpace instead of declaring a second copy of its matrix.
# An all-zero projection is rejected by LatentSpace, so importing this module raises until it is
# replaced — that is what keeps a half-filled declaration from reaching a generation.
{enum_name.upper()}_LATENT_RGB_FACTORS = [
    [0.0, 0.0, 0.0],
]

{enum_name.upper()}_LATENT_SPACE = LatentSpace(
    channels=1,  # TODO: latent channels; must equal len(...LATENT_RGB_FACTORS)
    spatial_compression=8,  # TODO: how much smaller a latent is than the image, per side
    rgb_factors={enum_name.upper()}_LATENT_RGB_FACTORS,
)

register(
    BaseModelType.{enum_name},
    LatentSpaceFacet({enum_name.upper()}_LATENT_SPACE),
    # TODO: the class its text encoder produces. Reuse an existing one where the shape matches.
    ConditioningFacet(BasicConditioningInfo),
    # TODO: what the generation sliders should say. Cite the model card in a comment.
    DefaultSettingsFacet({{None: MainModelDefaultSettings(width=1024, height=1024)}}),
    # TODO: which of txt2img / img2img / inpaint / outpaint / t2v / i2v it can do, and the prefix
    # its mode strings carry in image metadata. Both must match GENERATION_MODES.
    ModalityFacet(frozenset({{"txt2img"}}), metadata_slug="{base_value.replace("-", "_")}"),
    # TODO: dimension_grid must equal the `multiple_of` on this architecture's denoise node.
    FeaturesFacet(
        negative_prompt=NegativePrompt(visible=True, usage="always"),
        dimension_grid=8,
    ),
)
'''


def invocations_package(base_value: str, enum_name: str) -> str:
    return f'''"""{enum_name} nodes.

Node modules go here, keeping their architecture prefix: `{base_value.replace("-", "_")}_denoise.py`,
`{base_value.replace("-", "_")}_model_loader.py`. VAE, text-encoder and PiD nodes belong in
`invocations/vae/`, `invocations/text_encoder/` and `invocations/pid/` instead — they are shared
across architectures rather than owned by one.

This package is discovered automatically; there is no list to add it to. The `__init__.py` is what
makes it a package, and without it every node in here would silently not exist.
"""
'''


def starter_models_module(base_value: str, enum_name: str) -> str:
    return f'''"""{enum_name} starter models."""

from invokeai.backend.model_manager.starter_models.types import StarterModel  # noqa: F401

# TODO: declare the starter models, then add them to STARTER_MODELS in this package's __init__.py.
# That list is a curated order, not a derived one — insert where it belongs rather than appending.
'''


def derive_residual_edits(source_root: Path, threshold: int = 5) -> list[tuple[str, int]]:
    """Modules that dispatch on `BaseModelType`, worst first.

    The heuristic is deliberately crude and deliberately recomputed: a module naming many bases is
    choosing behaviour per base, and a new base has to be added to it.
    """
    found: list[tuple[str, int]] = []
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root.parent).as_posix()
        if relative.startswith(NOT_A_COST):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        members = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "BaseModelType"
        }
        if len(members) >= threshold:
            found.append((relative, len(members)))
    return sorted(found, key=lambda row: (-row[1], row[0]))


def planned_files(base_value: str, enum_name: str) -> dict[str, str]:
    slug = base_value.replace("-", "_")
    return {
        f"invokeai/backend/architectures/defs/{slug}.py": defs_module(base_value, enum_name),
        f"invokeai/app/invocations/{slug}/__init__.py": invocations_package(base_value, enum_name),
        f"invokeai/backend/model_manager/starter_models/{slug}.py": starter_models_module(base_value, enum_name),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="The BaseModelType value, e.g. 'new-model'")
    parser.add_argument("--name", required=True, help="The BaseModelType member name, e.g. 'NewModel'")
    parser.add_argument("--write", action="store_true", help="Write the files. Without this, print what would be.")
    args = parser.parse_args()

    files = planned_files(args.base, args.name)

    existing = [path for path in files if (REPO_ROOT / path).exists()]
    if existing:
        print("Refusing to overwrite:", *existing, sep="\n  ")
        return 1

    for path, content in files.items():
        if args.write:
            target = REPO_ROOT / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            print(f"wrote  {path}  ({content.count(chr(10))} lines)")
        else:
            print(f"would write  {path}  ({content.count(chr(10))} lines)")

    print(
        "\nThese cannot be scaffolded — the work is to add something, not to extend a dispatch:",
    )
    for path, what in UNDERIVABLE:
        print(f"  {path}\n      {what}")

    residual = derive_residual_edits(SOURCE_ROOT)
    print(f"\nAnd these {len(residual)} modules dispatch on BaseModelType, so check each one:")
    for path, count in residual:
        print(f"  {count:>3} bases   {path}")

    print(
        "\nThe app will not boot until the generated declaration is filled in — that is deliberate:"
        "\nits placeholder latent projection is all zeros, which LatentSpace refuses to construct."
        if args.write
        else "\nNothing was written. Re-run with --write."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
