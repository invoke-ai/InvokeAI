"""The scheduler family a variant is offered, where it differs from its architecture's.

webv2 used to special-case Z-Image Base in two places to drop LCM, against a `flow-no-lcm` set id the
backend never served. The rule is the node's -- `z_image_denoise` documents LCM as Turbo-only -- so
it is declared here, served on the variant row, and the frontend has no case of its own.
"""

from invokeai.app.invocations.z_image.z_image_denoise import ZImageDenoiseInvocation
from invokeai.backend.architectures import architecture_capabilities
from invokeai.backend.architectures.facets.features import FeaturesFacet
from invokeai.backend.architectures.registry import require
from invokeai.backend.model_manager.taxonomy import BaseModelType, ZImageVariantType


def test_z_image_base_is_offered_the_flow_set_without_lcm() -> None:
    # Read from the node rather than restated: if the node ever supports LCM on Base, this fails
    # and the declaration should go.
    description = ZImageDenoiseInvocation.model_fields["scheduler"].description or ""
    assert "LCM works with Turbo only (not Base)" in description

    facet = require(BaseModelType.ZImage, FeaturesFacet)
    assert facet.resolve_scheduler_set(ZImageVariantType.ZBase) == "flow-no-lcm"
    assert facet.resolve_scheduler_set(ZImageVariantType.Turbo) == "flow"
    assert facet.resolve_scheduler_set() == "flow"


def test_the_variant_set_is_what_the_table_serves() -> None:
    rows = {(row.base, row.variant): row for row in architecture_capabilities()}

    assert rows[(BaseModelType.ZImage, "zbase")].features.scheduler_set == "flow-no-lcm"
    assert rows[(BaseModelType.ZImage, None)].features.scheduler_set == "flow"
