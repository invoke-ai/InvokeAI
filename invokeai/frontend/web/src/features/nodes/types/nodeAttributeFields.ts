/**
 * Node attribute fields are declared on every invocation by the backend's `BaseInvocation`, but the workflow editor
 * treats them differently from ordinary inputs:
 *
 * - Their value lives on the node itself (`node.data.useCache` / `node.data.isIntermediate`), never in
 *   `node.data.inputs`. No field instance is ever created for them.
 * - They are parsed into invocation templates all the same, because a template entry is what connection validation
 *   and the connection handle need.
 * - They are filtered out of the node's input list and rendered in the node footer instead.
 */
const NODE_ATTRIBUTE_FIELD_NAMES = ['use_cache', 'is_intermediate'] as const;

export type NodeAttributeFieldName = (typeof NODE_ATTRIBUTE_FIELD_NAMES)[number];

export const isNodeAttributeFieldName = (fieldName: string): fieldName is NodeAttributeFieldName =>
  NODE_ATTRIBUTE_FIELD_NAMES.includes(fieldName as NodeAttributeFieldName);
