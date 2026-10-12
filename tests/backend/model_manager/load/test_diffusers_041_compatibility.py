import ast
import importlib
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import accelerate
import diffusers
import pytest
import torch
from packaging.version import Version

import invokeai
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader

_DiffusersImport = tuple[str, str, str]  # module, name, location


def _is_diffusers_import(node: ast.AST) -> bool:
    return isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").split(".")[0] == "diffusers"


def _is_version_gate(node: ast.If) -> bool:
    return any("version" in getattr(n, "id", getattr(n, "attr", "")) for n in ast.walk(node.test))


def _imports_in(statements: list[ast.stmt], path: Path) -> list[_DiffusersImport]:
    return [
        (node.module, alias.name, f"{path}:{node.lineno}")
        for statement in statements
        for node in ast.walk(statement)
        if _is_diffusers_import(node)
        for alias in node.names
    ]


def _collect_diffusers_imports(
    node: ast.AST, path: Path, plain: list[_DiffusersImport], gated: list[list[list[_DiffusersImport]]]
) -> None:
    """Collects `from diffusers... import X`, including function-local ones.

    A branch on a diffusers version only has to resolve on one side, so its two sides are kept as alternatives.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.If) and _is_version_gate(child):
            gated.append([_imports_in(child.body, path), _imports_in(child.orelse, path)])
            continue
        if _is_diffusers_import(child):
            plain.extend(_imports_in([child], path))
        _collect_diffusers_imports(child, path, plain, gated)


def _unresolved(imports: list[_DiffusersImport]) -> list[str]:
    failures = []
    for module_name, name, location in imports:
        try:
            module = importlib.import_module(module_name)
            if name != "*" and not hasattr(module, name):
                importlib.import_module(f"{module_name}.{name}")
        except ImportError as e:
            failures.append(f"{location}: from {module_name} import {name} ({e})")
    return failures


def test_every_diffusers_import_in_invokeai_resolves() -> None:
    # Most loaders import diffusers inside functions, so a module that moves upstream only fails when a user
    # loads that model. Resolving every import here catches it at the next bump instead.
    package_dir = Path(invokeai.__file__).parent
    plain: list[_DiffusersImport] = []
    gated: list[list[list[_DiffusersImport]]] = []
    for path in sorted(package_dir.rglob("*.py")):
        if "node_modules" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        _collect_diffusers_imports(tree, path.relative_to(package_dir), plain, gated)
    assert plain

    unresolved = _unresolved(plain)
    for branches in gated:
        failures = [_unresolved(branch) for branch in branches]
        if all(failures):
            unresolved.extend(failure for branch_failures in failures for failure in branch_failures)
    assert not unresolved, "\n".join(unresolved)


def test_pinned_diffusers_exposes_existing_and_krea_model_contracts() -> None:
    assert Version(diffusers.__version__) == Version("0.41.0")

    expected_symbols = (
        "AutoencoderKLFlux2",
        "FluxTransformer2DModel",
        "Flux2Transformer2DModel",
        "Krea2Transformer2DModel",
        "QwenImageTransformer2DModel",
        "StableDiffusionPipeline",
        "StableDiffusionXLPipeline",
        "ZImageTransformer2DModel",
    )
    for symbol in expected_symbols:
        assert getattr(diffusers, symbol, None) is not None, f"diffusers is missing {symbol}"


def test_flow_match_scheduler_keeps_custom_sigma_and_shift_api() -> None:
    parameters = signature(diffusers.FlowMatchEulerDiscreteScheduler.set_timesteps).parameters

    assert "sigmas" in parameters
    assert "mu" in parameters
    assert "device" in parameters


def test_invoke_generic_diffusers_loader_smoke(tmp_path) -> None:
    source_model = diffusers.AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(4,),
        layers_per_block=1,
        latent_channels=2,
        norm_num_groups=1,
        sample_size=8,
    )
    source_model.save_pretrained(tmp_path)

    loader = object.__new__(GenericDiffusersLoader)
    loader._torch_dtype = torch.float32
    loader._apply_fp8_layerwise_casting = lambda model, _config, _submodel: model

    loaded_model = loader._load_model(SimpleNamespace(path=str(tmp_path)))
    latents = loaded_model.encode(torch.zeros(1, 3, 8, 8)).latent_dist.mode()

    assert isinstance(loaded_model, diffusers.AutoencoderKL)
    assert latents.shape == (1, 2, 8, 8)


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda: diffusers.FluxTransformer2DModel(
                in_channels=4,
                num_layers=1,
                num_single_layers=1,
                attention_head_dim=4,
                num_attention_heads=1,
                joint_attention_dim=8,
                pooled_projection_dim=8,
                axes_dims_rope=(1, 1, 2),
            ),
            id="flux",
        ),
        pytest.param(
            lambda: diffusers.Flux2Transformer2DModel(
                in_channels=4,
                num_layers=1,
                num_single_layers=1,
                attention_head_dim=4,
                num_attention_heads=1,
                joint_attention_dim=8,
                timestep_guidance_channels=4,
                mlp_ratio=2,
                axes_dims_rope=(1, 1, 1, 1),
            ),
            id="flux2",
        ),
        pytest.param(
            lambda: diffusers.QwenImageTransformer2DModel(
                patch_size=1,
                in_channels=4,
                out_channels=4,
                num_layers=1,
                attention_head_dim=8,
                num_attention_heads=1,
                joint_attention_dim=8,
                axes_dims_rope=(2, 2, 4),
            ),
            id="qwen-image",
        ),
        pytest.param(
            lambda: diffusers.ZImageTransformer2DModel(
                all_patch_size=(1,),
                all_f_patch_size=(1,),
                in_channels=4,
                dim=8,
                n_layers=1,
                n_refiner_layers=1,
                n_heads=1,
                n_kv_heads=1,
                cap_feat_dim=8,
                axes_dims=[2, 2, 4],
                axes_lens=[8, 8, 8],
            ),
            id="z-image",
        ),
        pytest.param(
            lambda: diffusers.Krea2Transformer2DModel(
                in_channels=4,
                num_layers=1,
                attention_head_dim=8,
                num_attention_heads=1,
                num_key_value_heads=1,
                intermediate_size=16,
                timestep_embed_dim=8,
                text_hidden_dim=8,
                num_text_layers=2,
                text_num_attention_heads=1,
                text_num_key_value_heads=1,
                text_intermediate_size=16,
                num_layerwise_text_blocks=1,
                num_refiner_text_blocks=1,
                axes_dims_rope=(2, 2, 4),
            ),
            id="krea-2",
        ),
        pytest.param(
            lambda: diffusers.AutoencoderKLWan(
                base_dim=4,
                decoder_base_dim=4,
                z_dim=4,
                dim_mult=[1, 1],
                num_res_blocks=1,
                temperal_downsample=[False],
                latents_mean=[0.0] * 4,
                latents_std=[1.0] * 4,
                scale_factor_temporal=1,
                scale_factor_spatial=2,
            ),
            id="anima-vae",
        ),
    ],
)
def test_pinned_diffusers_constructs_representative_transformer_and_vae_configs(factory) -> None:
    with accelerate.init_empty_weights():
        model = factory()

    assert model is not None


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda: diffusers.StableDiffusionPipeline(
                vae=None,
                text_encoder=None,
                tokenizer=None,
                unet=None,
                scheduler=None,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            ),
            id="stable-diffusion",
        ),
        pytest.param(
            lambda: diffusers.StableDiffusionXLPipeline(
                vae=None,
                text_encoder=None,
                text_encoder_2=None,
                tokenizer=None,
                tokenizer_2=None,
                unet=None,
                scheduler=None,
                add_watermarker=False,
            ),
            id="stable-diffusion-xl",
        ),
    ],
)
def test_pinned_diffusers_constructs_existing_stable_diffusion_pipelines(factory) -> None:
    pipeline = factory()

    assert pipeline is not None
