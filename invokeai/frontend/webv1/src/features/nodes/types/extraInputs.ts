/**
 * Some backend nodes are configured with pydantic `extra='allow'` and accept inputs that aren't
 * declared in the OpenAPI schema. The most important case is `core_metadata`, which collects
 * generation-mode-specific recall data (`z_image_seed_variance_*`, `dype_preset`, `ref_images`, ...).
 *
 * The frontend must round-trip those extras intact:
 *   - `graphToWorkflow` synthesizes a `MetadataExtraField` template for them
 *   - `buildNodesGraph` forwards their values verbatim
 *   - `fieldValidators` does not treat them as errors
 *   - `InputFieldGate` does not render an "unexpected field" warning
 *
 * See: https://github.com/invoke-ai/InvokeAI/issues/9151
 */
const NODES_ACCEPTING_EXTRA_INPUTS: ReadonlySet<string> = new Set(['core_metadata']);

export const nodeAcceptsExtraInputs = (nodeType: string): boolean => NODES_ACCEPTING_EXTRA_INPUTS.has(nodeType);
