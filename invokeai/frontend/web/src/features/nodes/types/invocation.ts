import type { Edge, Node } from '@xyflow/react';
import { z } from 'zod';

import { zClassification, zProgressImage } from './common';
import { nodeAcceptsExtraInputs } from './extraInputs';
import type { FieldInputInstance } from './field';
import { zFieldInputInstance, zFieldInputInstanceWithExtras, zFieldInputTemplate, zFieldOutputTemplate } from './field';
import { zSemVer } from './semver';

// #region InvocationTemplate
const _zInvocationTemplate = z.object({
  type: z.string(),
  title: z.string(),
  description: z.string(),
  tags: z.array(z.string().min(1)),
  inputs: z.record(z.string(), zFieldInputTemplate),
  outputs: z.record(z.string(), zFieldOutputTemplate),
  outputType: z.string().min(1),
  version: zSemVer,
  useCache: z.boolean(),
  nodePack: z.string().min(1).default('invokeai'),
  classification: zClassification,
  category: z.string().default('other'),
});
export type InvocationTemplate = z.infer<typeof _zInvocationTemplate>;
// #endregion

// #region NodeData
export const zInvocationNodeData = z
  .object({
    id: z.string().trim().min(1),
    version: zSemVer,
    nodePack: z.string().min(1).default('invokeai'),
    label: z.string(),
    notes: z.string(),
    type: z.string().trim().min(1),
    // Parsed per-input in the transform below so that the input-instance schema can be chosen based
    // on the node type (extras are only accepted for nodes that declare `extra='allow'`).
    inputs: z.record(z.string(), z.unknown()),
    isOpen: z.boolean(),
    isIntermediate: z.boolean(),
    useCache: z.boolean(),
  })
  .transform((data, ctx) => {
    // Undeclared "extra" inputs (pydantic `extra='allow'`, e.g. `core_metadata`) carry arbitrary
    // values that must round-trip. Only nodes known to accept extras get the permissive
    // `MetadataExtraField` catch-all; every other node uses the strict instance union so stale or
    // malformed saved values are coerced away (via the stateless branch) rather than preserved and
    // later leaked into the backend graph by `buildNodesGraph`.
    const instanceSchema = nodeAcceptsExtraInputs(data.type) ? zFieldInputInstanceWithExtras : zFieldInputInstance;
    const inputs: Record<string, FieldInputInstance> = {};
    for (const [name, rawInput] of Object.entries(data.inputs)) {
      const result = instanceSchema.safeParse(rawInput);
      if (!result.success) {
        ctx.addIssue({
          code: 'custom',
          message: `Invalid input "${name}": ${result.error.message}`,
          path: ['inputs', name],
        });
        return z.NEVER;
      }
      inputs[name] = result.data;
    }
    return { ...data, inputs };
  });

export const zNotesNodeData = z.object({
  id: z.string().trim().min(1),
  type: z.literal('notes'),
  label: z.string(),
  isOpen: z.boolean(),
  notes: z.string(),
});
export const zConnectorNodeData = z.object({
  id: z.string().trim().min(1),
  type: z.literal('connector'),
  label: z.string(),
  isOpen: z.boolean(),
});
const zCurrentImageNodeData = z.object({
  id: z.string().trim().min(1),
  type: z.literal('current_image'),
  label: z.string(),
  isOpen: z.boolean(),
});

export type NotesNodeData = z.infer<typeof zNotesNodeData>;
export type InvocationNodeData = z.infer<typeof zInvocationNodeData>;
export type ConnectorNodeData = z.infer<typeof zConnectorNodeData>;
type CurrentImageNodeData = z.infer<typeof zCurrentImageNodeData>;

const zInvocationNodeValidationSchema = z.looseObject({
  type: z.literal('invocation'),
  data: zInvocationNodeData,
});
const zInvocationNode = z.custom<Node<InvocationNodeData, 'invocation'>>(
  (val) => zInvocationNodeValidationSchema.safeParse(val).success
);
export type InvocationNode = z.infer<typeof zInvocationNode>;

const zNotesNodeValidationSchema = z.looseObject({
  type: z.literal('notes'),
  data: zNotesNodeData,
});
const zNotesNode = z.custom<Node<NotesNodeData, 'notes'>>((val) => zNotesNodeValidationSchema.safeParse(val).success);
export type NotesNode = z.infer<typeof zNotesNode>;

const zConnectorNodeValidationSchema = z.looseObject({
  type: z.literal('connector'),
  data: zConnectorNodeData,
});
const zConnectorNode = z.custom<Node<ConnectorNodeData, 'connector'>>(
  (val) => zConnectorNodeValidationSchema.safeParse(val).success
);
export type ConnectorNode = z.infer<typeof zConnectorNode>;

const zCurrentImageNodeValidationSchema = z.looseObject({
  type: z.literal('current_image'),
  data: zCurrentImageNodeData,
});
const zCurrentImageNode = z.custom<Node<CurrentImageNodeData, 'current_image'>>(
  (val) => zCurrentImageNodeValidationSchema.safeParse(val).success
);
export type CurrentImageNode = z.infer<typeof zCurrentImageNode>;

export const zAnyNode = z.union([zInvocationNode, zNotesNode, zConnectorNode, zCurrentImageNode]);
export type AnyNode = z.infer<typeof zAnyNode>;

export const isInvocationNode = (node?: AnyNode | null): node is InvocationNode =>
  Boolean(node && node.type === 'invocation');
export const isNotesNode = (node?: AnyNode | null): node is NotesNode => Boolean(node && node.type === 'notes');
export const isConnectorNode = (node?: AnyNode | null): node is ConnectorNode =>
  Boolean(node && node.type === 'connector');
// #endregion

// #region NodeExecutionState
export const zNodeStatus = z.enum(['PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED']);
const zNodeError = z.object({
  error_type: z.string(),
  error_message: z.string(),
  error_traceback: z.string(),
});
const _zNodeExecutionState = z.object({
  nodeId: z.string().trim().min(1),
  status: zNodeStatus,
  progress: z.number().nullable(),
  progressImage: zProgressImage.nullable(),
  outputs: z.array(z.any()),
  error: zNodeError.nullable(),
});
export type NodeExecutionState = z.infer<typeof _zNodeExecutionState>;
// #endregion

// #region Edges
const zDefaultInvocationNodeEdgeValidationSchema = z.looseObject({
  type: z.literal('default'),
});
const zDefaultInvocationNodeEdge = z.custom<Edge<Record<string, never>, 'default'>>(
  (val) => zDefaultInvocationNodeEdgeValidationSchema.safeParse(val).success
);
export type DefaultInvocationNodeEdge = z.infer<typeof zDefaultInvocationNodeEdge>;

const zInvocationNodeEdgeCollapsedData = z.object({
  count: z.number().int().min(1),
});
const zInvocationNodeEdgeCollapsedValidationSchema = z.looseObject({
  type: z.literal('default'),
  data: zInvocationNodeEdgeCollapsedData,
});
type InvocationNodeEdgeCollapsedData = z.infer<typeof zInvocationNodeEdgeCollapsedData>;

const zCollapsedInvocationNodeEdge = z.custom<Edge<InvocationNodeEdgeCollapsedData, 'collapsed'>>(
  (val) => zInvocationNodeEdgeCollapsedValidationSchema.safeParse(val).success
);
export type CollapsedInvocationNodeEdge = z.infer<typeof zCollapsedInvocationNodeEdge>;
export const zAnyEdge = z.union([zDefaultInvocationNodeEdge, zCollapsedInvocationNodeEdge]);
export type AnyEdge = z.infer<typeof zAnyEdge>;
// #endregion

export const isBatchNodeType = (type: string) =>
  ['image_batch', 'string_batch', 'integer_batch', 'float_batch'].includes(type);

export const isGeneratorNodeType = (type: string) =>
  ['image_generator', 'string_generator', 'integer_generator', 'float_generator'].includes(type);

export const isBatchNode = (node: InvocationNode) => isBatchNodeType(node.data.type);
const isGeneratorNode = (node: InvocationNode) => isGeneratorNodeType(node.data.type);

export const isExecutableNode = (node: InvocationNode) => {
  return !isBatchNode(node) && !isGeneratorNode(node);
};
