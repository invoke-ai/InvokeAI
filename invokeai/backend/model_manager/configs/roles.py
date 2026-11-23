"""Model roles: the other model types a record's files serve as.

A record is filed under one primary `type`. Identification adds `roles` for the other `(type, format, variant)` its
files load as, so a consumer asks `serves_as` instead of requiring one type. A loader that needs the role's config
class gets `view_as`.
"""

from pydantic import ValidationError

from invokeai.backend.model_manager.configs.base import ModelRole
from invokeai.backend.model_manager.configs.factory import AnyModelConfig, ModelConfigFactory
from invokeai.backend.model_manager.taxonomy import AnyVariant, ModelType


class ModelRoleError(ValueError):
    """A model asked for in a role it does not serve."""


def serves_as(config: AnyModelConfig, model_type: ModelType) -> bool:
    """Whether the model is of this type, as filed or by one of its roles."""
    return config.type == model_type or any(role.type == model_type for role in config.roles)


def role_variant(config: AnyModelConfig, model_type: ModelType) -> AnyVariant | None:
    """The variant the model serves `model_type` in, or None if it has none or does not serve it."""
    if config.type == model_type:
        return getattr(config, "variant", None)
    return next((role.variant for role in config.roles if role.type == model_type), None)


def view_as(config: AnyModelConfig, model_type: ModelType) -> AnyModelConfig:
    """The config of the model in the role `model_type`: an instance of that role's config class, same files.

    The record's own config when it is already filed under `model_type`. The view is not stored; it lists the record's
    primary type among its roles and `role_of` names that type.
    """
    if config.type == model_type:
        return config
    role = next((r for r in config.roles if r.type == model_type), None)
    if role is None:
        raise ModelRoleError(f"model '{config.name}' is a {config.type.value} and does not serve as {model_type.value}")

    fields = config.model_dump()
    fields.pop("variant", None)
    fields["type"] = role.type
    fields["format"] = role.format or config.format
    if role.variant is not None:
        fields["variant"] = role.variant
    fields["roles"] = [r for r in config.roles if r.type != model_type] + [
        ModelRole(
            type=config.type,
            format=config.format if config.format != fields["format"] else None,
            variant=getattr(config, "variant", None),
        )
    ]
    try:
        view = ModelConfigFactory.from_dict(fields)
    except ValidationError as error:
        raise ModelRoleError(
            f"model '{config.name}' cannot be read as {model_type.value}: its role is incomplete; re-identify it"
        ) from error
    view._role_of = config.type
    return view
