"""FastAPI routes for custom node management."""

import json
import shutil
import subprocess
import sys
import traceback
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Optional

from fastapi import Body
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.invocations.baseinvocation import InvocationRegistry
from invokeai.app.services.config.config_default import get_config
from invokeai.app.services.workflow_records.workflow_records_common import WorkflowWithoutIDValidator
from invokeai.backend.util.logging import InvokeAILogger

custom_nodes_router = APIRouter(prefix="/v2/custom_nodes", tags=["custom_nodes"])

logger = InvokeAILogger.get_logger()


class NodePackInfo(BaseModel):
    """Information about an installed node pack."""

    name: str = Field(description="The name of the node pack.")
    path: str = Field(description="The path to the node pack directory.")
    node_count: int = Field(description="The number of nodes in the pack.")
    node_types: list[str] = Field(description="The invocation types provided by this node pack.")


class NodePackListResponse(BaseModel):
    """Response for listing installed node packs."""

    node_packs: list[NodePackInfo] = Field(description="List of installed node packs.")


class InstallNodePackRequest(BaseModel):
    """Request to install a node pack from a git URL."""

    source: str = Field(description="Git URL of the node pack to install.")


class InstallNodePackResponse(BaseModel):
    """Response after installing a node pack."""

    name: str = Field(description="The name of the installed node pack.")
    success: bool = Field(description="Whether the installation was successful.")
    message: str = Field(description="Status message.")
    workflows_imported: int = Field(default=0, description="Number of workflows imported from the pack.")
    requires_dependencies: bool = Field(
        default=False,
        description="Whether the pack ships a dependency manifest (requirements.txt or pyproject.toml) "
        "that the user must install manually following the pack's documentation.",
    )
    dependency_file: Optional[str] = Field(
        default=None,
        description="Name of the detected dependency manifest file, if any.",
    )


class UninstallNodePackResponse(BaseModel):
    """Response after uninstalling a node pack."""

    name: str = Field(description="The name of the uninstalled node pack.")
    success: bool = Field(description="Whether the uninstall was successful.")
    message: str = Field(description="Status message.")


def _get_custom_nodes_path() -> Path:
    """Returns the configured custom nodes directory path."""
    config = get_config()
    return config.custom_nodes_path


def _get_installed_packs() -> list[NodePackInfo]:
    """Scans the custom nodes directory and returns info about installed packs."""
    custom_nodes_path = _get_custom_nodes_path()

    if not custom_nodes_path.exists():
        return []

    packs: list[NodePackInfo] = []

    # Get all node types grouped by node_pack
    node_types_by_pack: dict[str, list[str]] = {}
    for inv_class in InvocationRegistry._invocation_classes:
        node_pack = inv_class.UIConfig.node_pack
        inv_type = inv_class.get_type()
        if node_pack not in node_types_by_pack:
            node_types_by_pack[node_pack] = []
        node_types_by_pack[node_pack].append(inv_type)

    for d in sorted(custom_nodes_path.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("_") or d.name.startswith("."):
            continue
        init = d / "__init__.py"
        if not init.exists():
            continue

        pack_name = d.name
        node_types = node_types_by_pack.get(pack_name, [])

        packs.append(
            NodePackInfo(
                name=pack_name,
                path=str(d),
                node_count=len(node_types),
                node_types=node_types,
            )
        )

    return packs


@custom_nodes_router.get(
    "/",
    operation_id="list_custom_node_packs",
    response_model=NodePackListResponse,
)
async def list_custom_node_packs() -> NodePackListResponse:
    """Lists all installed custom node packs."""
    packs = _get_installed_packs()
    return NodePackListResponse(node_packs=packs)


@custom_nodes_router.post(
    "/install",
    operation_id="install_custom_node_pack",
    response_model=InstallNodePackResponse,
)
async def install_custom_node_pack(
    request: InstallNodePackRequest = Body(description="The source URL to install from."),
) -> InstallNodePackResponse:
    """Installs a custom node pack from a git URL by cloning it into the nodes directory."""
    custom_nodes_path = _get_custom_nodes_path()
    custom_nodes_path.mkdir(parents=True, exist_ok=True)

    source = request.source.strip()

    # Extract pack name from URL
    pack_name = source.rstrip("/").split("/")[-1]
    if pack_name.endswith(".git"):
        pack_name = pack_name[:-4]

    target_dir = custom_nodes_path / pack_name

    if target_dir.exists():
        return InstallNodePackResponse(
            name=pack_name,
            success=False,
            message=f"Node pack '{pack_name}' already exists. Uninstall it first to reinstall.",
        )

    try:
        # Clone the repository
        result = subprocess.run(
            ["git", "clone", source, str(target_dir)],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            # Clean up on failure
            if target_dir.exists():
                shutil.rmtree(target_dir)
            return InstallNodePackResponse(
                name=pack_name,
                success=False,
                message=f"Git clone failed: {result.stderr.strip()}",
            )

        # Detect dependency manifests but do NOT install them automatically.
        # The user is responsible for installing dependencies per the pack's documentation,
        # since arbitrary pip installs can break the InvokeAI environment.
        dependency_file: Optional[str] = None
        for candidate in ("requirements.txt", "pyproject.toml"):
            if (target_dir / candidate).exists():
                dependency_file = candidate
                logger.info(f"Node pack '{pack_name}' ships a {candidate}; user must install dependencies manually.")
                break

        # Check for __init__.py
        init_file = target_dir / "__init__.py"
        if not init_file.exists():
            shutil.rmtree(target_dir)
            return InstallNodePackResponse(
                name=pack_name,
                success=False,
                message=f"Node pack '{pack_name}' does not contain an __init__.py file.",
            )

        # Load the node pack at runtime
        _load_node_pack(pack_name, target_dir)

        # Import any workflows found in the pack
        workflows_imported = _import_workflows_from_pack(target_dir, pack_name)
        workflow_msg = f" Imported {workflows_imported} workflow(s)." if workflows_imported > 0 else ""
        dependency_msg = (
            f" This pack includes a {dependency_file} — install its dependencies manually following the pack's documentation."
            if dependency_file
            else ""
        )

        return InstallNodePackResponse(
            name=pack_name,
            success=True,
            message=f"Successfully installed node pack '{pack_name}'.{workflow_msg}{dependency_msg}",
            workflows_imported=workflows_imported,
            requires_dependencies=dependency_file is not None,
            dependency_file=dependency_file,
        )

    except subprocess.TimeoutExpired:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        return InstallNodePackResponse(
            name=pack_name,
            success=False,
            message="Installation timed out.",
        )
    except Exception:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        error = traceback.format_exc()
        logger.error(f"Failed to install node pack {pack_name}: {error}")
        return InstallNodePackResponse(
            name=pack_name,
            success=False,
            message=f"Installation failed: {error}",
        )


@custom_nodes_router.delete(
    "/{pack_name}",
    operation_id="uninstall_custom_node_pack",
    response_model=UninstallNodePackResponse,
)
async def uninstall_custom_node_pack(pack_name: str) -> UninstallNodePackResponse:
    """Uninstalls a custom node pack by removing its directory.

    Note: A restart is required for the node removal to take full effect.
    Installed nodes from the pack will remain registered until restart.
    """
    custom_nodes_path = _get_custom_nodes_path()
    target_dir = custom_nodes_path / pack_name

    if not target_dir.exists():
        return UninstallNodePackResponse(
            name=pack_name,
            success=False,
            message=f"Node pack '{pack_name}' not found.",
        )

    try:
        shutil.rmtree(target_dir)

        # Unregister the nodes from the registry so they disappear immediately
        removed_types = InvocationRegistry.unregister_pack(pack_name)
        if removed_types:
            # Invalidate OpenAPI schema cache so frontend gets updated node definitions
            from invokeai.app.api_app import app

            app.openapi_schema = None
            logger.info(
                f"Unregistered {len(removed_types)} node(s) from pack '{pack_name}': {', '.join(removed_types)}"
            )

        # Remove the module from sys.modules so it won't interfere
        if pack_name in sys.modules:
            del sys.modules[pack_name]

        # Remove workflows that were imported from this node pack
        workflows_removed = _remove_workflows_for_pack(pack_name)
        workflow_msg = f" Removed {workflows_removed} workflow(s)." if workflows_removed > 0 else ""

        return UninstallNodePackResponse(
            name=pack_name,
            success=True,
            message=f"Successfully uninstalled node pack '{pack_name}'.{workflow_msg}",
        )
    except Exception:
        error = traceback.format_exc()
        logger.error(f"Failed to uninstall node pack {pack_name}: {error}")
        return UninstallNodePackResponse(
            name=pack_name,
            success=False,
            message=f"Uninstall failed: {error}",
        )


@custom_nodes_router.post(
    "/reload",
    operation_id="reload_custom_nodes",
)
async def reload_custom_nodes() -> dict[str, str]:
    """Triggers a reload of all custom nodes.

    This re-scans the nodes directory and loads any new node packs.
    Already loaded packs are skipped.
    """
    config = get_config()
    custom_nodes_path = config.custom_nodes_path

    if not custom_nodes_path.exists():
        return {"status": "No custom nodes directory found."}

    from invokeai.app.invocations.load_custom_nodes import load_custom_nodes

    load_custom_nodes(custom_nodes_path, logger)

    # Invalidate the OpenAPI schema cache so the frontend gets updated node definitions
    from invokeai.app.api_app import app

    app.openapi_schema = None

    return {"status": "Custom nodes reloaded successfully."}


def _load_node_pack(pack_name: str, pack_dir: Path) -> None:
    """Loads a single node pack at runtime."""
    init = pack_dir / "__init__.py"
    if not init.exists():
        return

    if pack_name in sys.modules:
        logger.info(f"Node pack {pack_name} already loaded, skipping.")
        return

    spec = spec_from_file_location(pack_name, init.absolute())
    if spec is None or spec.loader is None:
        logger.warning(f"Could not load {init}")
        return

    logger.info(f"Loading node pack {pack_name}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # Invalidate OpenAPI schema cache
    from invokeai.app.api_app import app

    app.openapi_schema = None

    logger.info(f"Successfully loaded node pack {pack_name}")


def _import_workflows_from_pack(pack_dir: Path, pack_name: str) -> int:
    """Scans a node pack directory for workflow JSON files and imports them into the workflow library.

    A JSON file is considered a workflow if it contains 'nodes' and 'edges' keys at the top level.
    Workflows are imported as user workflows with the node pack name added to the tags.

    Returns the number of workflows successfully imported.
    """
    imported_count = 0

    # Search for .json files recursively
    for json_file in pack_dir.rglob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check if this looks like a workflow (must have nodes and edges)
            if not isinstance(data, dict):
                continue
            if "nodes" not in data or "edges" not in data:
                continue

            # Ensure the workflow has a meta section with category set to "user"
            if "meta" not in data:
                data["meta"] = {"version": "3.0.0", "category": "user"}
            else:
                data["meta"]["category"] = "user"

            # Add the node pack name to tags for discoverability
            existing_tags = data.get("tags", "")
            pack_tag = f"node-pack:{pack_name}"
            if pack_tag not in existing_tags:
                data["tags"] = f"{existing_tags}, {pack_tag}".strip(", ") if existing_tags else pack_tag

            # Remove the 'id' field if present — the system will assign a new one
            data.pop("id", None)

            # Validate and import the workflow
            workflow = WorkflowWithoutIDValidator.validate_python(data)
            ApiDependencies.invoker.services.workflow_records.create(workflow=workflow)
            logger.info(f"Imported workflow '{workflow.name}' from node pack '{pack_name}'")
            imported_count += 1

        except Exception:
            logger.warning(f"Skipped non-workflow or invalid JSON file: {json_file}")
            continue

    if imported_count > 0:
        logger.info(f"Imported {imported_count} workflow(s) from node pack '{pack_name}'")

    return imported_count


def _remove_workflows_for_pack(pack_name: str) -> int:
    """Removes all workflows that were imported from a node pack.

    Finds workflows tagged with 'node-pack:<pack_name>' and deletes them.
    Returns the number of workflows removed.
    """
    from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
    from invokeai.app.services.workflow_records.workflow_records_common import (
        WorkflowCategory,
        WorkflowRecordOrderBy,
    )

    pack_tag = f"node-pack:{pack_name}"
    removed_count = 0

    try:
        # Find all workflows with this pack's tag
        result = ApiDependencies.invoker.services.workflow_records.get_many(
            order_by=WorkflowRecordOrderBy.Name,
            direction=SQLiteDirection.Ascending,
            categories=[WorkflowCategory.User],
            tags=[pack_tag],
        )

        for workflow_item in result.items:
            try:
                ApiDependencies.invoker.services.workflow_records.delete(workflow_item.workflow_id)
                logger.info(f"Removed workflow '{workflow_item.name}' (from node pack '{pack_name}')")
                removed_count += 1
            except Exception:
                logger.warning(f"Failed to remove workflow '{workflow_item.name}'")

    except Exception:
        logger.warning(f"Failed to query workflows for node pack '{pack_name}'")

    if removed_count > 0:
        logger.info(f"Removed {removed_count} workflow(s) from node pack '{pack_name}'")

    return removed_count
